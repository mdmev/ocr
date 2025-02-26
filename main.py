import argparse
import os
import random
import csv
import time
import logging
import json

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
    sources_traductor = args.sources_traductor

    sources_traductor = get_list(sources_traductor)

    ## Load the database ##
    database = Mongo(connection_uri=os.getenv("MONGO_URI"))

    # ## Load the model ##
    logger.info("Initializing model...")

    for idx, file_path in enumerate(tqdm(sources_traductor, desc="Processing sources-traductor images")):
        print("#"*50)
        image_name = os.path.basename(file_path)
        logger.debug(f"Processing image: {image_name}")
        image = Image_(path=file_path)


if __name__ == "__main__":
    main()
