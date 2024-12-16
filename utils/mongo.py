import pymongo
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class Mongo:
    def __init__(self, connection_uri: str):
        self.client = pymongo.MongoClient(connection_uri)
        self._test_connection()

    def extract_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        logger.debug("Extracting metadata for image: %s", image_path)
        image_name = image_path.split('/')[-1]
        document = self.client.data_repository.files.find_one({"blob_filename": image_name})
        if not document:
            logger.warning("Image '%s' not found in the database.", image_name)
            return None
        
        rotation_key = document['best_metadata']['rotation']
        rotation = document['detections_history'][rotation_key]['rotation']

        corners_key = document['best_metadata']['corners']
        corners = document['detections_history'][corners_key]['corners']

        return {"rotation": rotation, "corners": corners}

    def _test_connection(self) -> None:
        try:
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful.")
        except Exception as e:
            logger.exception("Connection to MongoDB failed: %s", e)
            raise