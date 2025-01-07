import argparse
import os
import random
import csv
import time
import logging
import shutil
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2
import supervision as sv
from PIL import Image

from tqdm import tqdm
from typing import List, Dict, Any, Optional

from models.claude import ClaudeAPI
from models.gpt import OpenAIAPI
from models.yolo import YoloSegment
from utils.image import Image_
from utils.mongo import Mongo
from utils.utils import *
from utils.args import Arguments
from utils.bounding_box_matcher import BoundingBoxMatcher
from utils.bounding_box_cropper import BoundingBoxCropper

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/test.log"),
    ]
)
logger = logging.getLogger(__name__)

def main() -> None:
    ## Load the arguments ##
    args = Arguments.parse_arguments()
    model = args.model
    dataset = args.dataset
    output = args.output

    output_data = load_json(output)
    
    ## Load the images ##
    images = get_list(dataset)

    # ## Load the model ##
    logger.info("Initializing model...")
    if model == "sonnet":
        model = ClaudeAPI(model_name="claude-3-5-sonnet-20241022", prompt_type="test", tool="extract_json")
        cost = model.calculate_cost()
        logger.info(f"Initial cost: {cost}")


    yolo_segmenter = YoloSegment(
        model_path="weights/yolov11x_best.pt",
        conf_threshold=0.2,
        iou_threshold=0.8
    )

    matcher = BoundingBoxMatcher(distance_threshold=50.0)
    cont = 0
    for idx, file_path in enumerate(tqdm(images, desc="Processing images")):
        if idx >= 20:
            break

        image_name = os.path.basename(file_path)
        if image_name not in output_data:
                output_data[image_name] = []

        logger.debug(f"Processing image: {image_name}")
        image = Image_(path=file_path)
        left, right = image.split()
        left = left.to_cv2()
        right = right.to_cv2()

        left_detections = yolo_segmenter.run_inference(left)
        right_detections = yolo_segmenter.run_inference(right)
        left_detections = yolo_segmenter.filter_text_detections(left_detections, exclude_classes=[6])
        right_detections = yolo_segmenter.filter_text_detections(right_detections, exclude_classes=[6])

        matches_by_class = matcher.match(left_detections, right_detections)

        cropper = BoundingBoxCropper()
        boxes = cropper.crop_boxes_and_display(
            left,
            right,
            matches_by_class,
            left_detections,
            right_detections
        )


        for box in boxes:
            left_crop, right_crop, _, _ = box

            left_crop_pil = Image.fromarray(cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB))
            right_crop_pil = Image.fromarray(cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB))

            left_image = Image_(image=left_crop_pil)
            right_image = Image_(image=right_crop_pil)

            model.set_prompt("extract_raw_spanish")
            spanish = model.run_inference(left_image, max_retries=1, timeout=15)
            model.set_prompt("extract_raw_nahuatl")
            nahuatl = model.run_inference(right_image, max_retries=1, timeout=15)
            cont += 1

            output_data[image_name].append({
                "spanish": spanish,
                "nahuatl": nahuatl
            })
            save_json(output, output_data)

            logger.info(f"Current pairs: {cont}")
        cost = model.calculate_cost()
        logger.info(f"Current cost: {cost}")

if __name__ == "__main__":
    main()
