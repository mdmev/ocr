import argparse
import os
import random
import csv
import time
import logging

import pandas as pd

from tqdm import tqdm
from typing import List, Dict, Any, Optional

from models.claude import ClaudeAPI
from models.gpt import OpenAIAPI
from utils.image import Image_
from utils.mongo import Mongo
from utils.utils import *
from utils.args import Arguments

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/price_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main() -> None:
    ## Load the arguments ##
    args = Arguments.parse_arguments()
    model = args.model
    ground_truth_data = args.ground_truth_data
    sources_traductor = args.sources_traductor
    # output_csv = args.output_csv

    # output = pd.read_csv(output_csv)
    # if output.empty:
    #     output = pd.DataFrame(columns=["image", "candidate", "json"])



    gt_data = pd.read_csv(ground_truth_data)
    prefix = "/home/guillfa/CENIA/sources-traductor/"
    gt_data['image'] = gt_data['image'].apply(lambda x: prefix + x)    
    images_folder = gt_data['image'].tolist()

    sources_traductor = get_list(sources_traductor)
    sources_traductor = [img for img in sources_traductor if os.path.basename(img) not in {os.path.basename(c) for c in images_folder}]
    random.shuffle(sources_traductor)
    sources_traductor = sources_traductor[:300]

    # images = images_folder + sources_traductor
    images = images_folder
    ## Load the database ##
    database = Mongo(connection_uri=os.getenv("MONGO_URI"))

    # ## Load the model ##
    logger.info("Initializing model...")
    if model == "sonnet":
        model = ClaudeAPI(model_name="claude-3-5-sonnet-20241022", prompt_type="Classifier", tool="extract_json")
        cost = model.calculate_cost()
        logger.info(f"Initial cost: {cost}")

    processed_images = {
        "pairs_candidates": {},
        "sources_traductor": {}
    }



    for idx, file_path in enumerate(tqdm(images, desc="Processing sources-traductor images")):
        print("#"*50)
        if idx >= 50:
            break
        image_name = os.path.basename(file_path)
        logger.debug(f"Processing image: {image_name}")
        # if image_name in output['image'].values:
        #     logger.info(f"Image '{image_name}' already processed. Skipping...")
        #     continue

        image = Image_(path=file_path)
        image.resize_aspect_ratio()

        class_answer = model.run_inference(image, max_retries=3, timeout=15)
        cost = model.calculate_cost()
        logger.info(f"Current cost: {cost}")

        if class_answer != "Candidate":
            logger.info(f"Image '{image_name}' is not a candidate. Continuing...")
            continue

        model.set_prompt("JSON_extractor")
        answer = model.run_inference(image, max_retries=3, timeout=15)
        model.set_prompt("Classifier")
        cost = model.calculate_cost()
        logger.info(f"Current cost: {cost}")
        
        if model.total_cost > 30:
            logger.warning("Cost limit reached. Exiting.")
            break

        processed_images["pairs_candidates"][file_path.split('/')[-1]] = {
            "candidate": class_answer,
            "json": answer
        }
        logger.info(f"Saving results for '{image_name}' to CSV.")
        print(f"pairs_candidates: {file_path.split('/')[-1]}")
        print("candidate: ", class_answer)
        print("json: ", answer)

    
        # save_to_csv(
        #     output_csv, {"ground_truth": {file_path.split('/')[-1]: 
        #                                        {"candidate": class_answer,
        #                                         "json": answer
        #                                         }
        #                                     }
        #                 }
        #             )    
    # logger.info("Calculating metrics...")    
    # Calculate metrics (Can be improved)
    # calculate_metrics(output_csv)

    
if __name__ == "__main__":
    main()
