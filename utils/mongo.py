import pymongo
import logging
import json
from typing import Optional, Dict, Any, List
from pymongo import UpdateOne, MongoClient
from bson.objectid import ObjectId

logger = logging.getLogger(__name__)

class Mongo:
    def __init__(self, connection_uri: str):
        self.client = pymongo.MongoClient(connection_uri)
        self.collection = self.client.data_repository.files
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

    def update_rotation_bulk(self, updates: List[Dict[str, Any]]) -> None:
        if not updates:
            logger.warning("No updates to perform.")
            return
        
        bulk_updates = [
            self._update_rotation_query(
                update["current_mongo_element"],
                update["new_rotation"],
                update["technique_used"],
                update["confidence"],
                update["is_new_best"]
            ) for update in updates
        ]

        if bulk_updates:
            try:
                result = self.collection.bulk_write(bulk_updates, ordered=False)
                logger.info("Bulk update completed: Matched: %d, Modified: %d", result.matched_count, result.modified_count)
            except pymongo.errors.PyMongoError as e:
                logger.exception("Bulk update failed: %s", e)
                raise

    def update_corners_bulk(self, json_path: str) -> None:
        try:
            with open(json_path) as f:
                inferences_data = json.load(f)
        except Exception as e:
            logger.error("Failed to read JSON file: %s", e)
            raise

        bulk_updates = []

        for item in inferences_data:
            filename = item.get("filename")
            corners = item.get("corners", [])

            if not filename:
                continue

            document = self.collection.find_one({"blob_filename": filename})

            if document:
                bulk_updates.append(
                    self._update_corners_query(
                        document,
                        corners,
                        "SegFormerb5 versión febrero 2025",
                        "v1",  # Hardcoded for now, can be improved later
                        True  # is_new_best
                    )
                )

        if bulk_updates:
            try:
                result = self.collection.bulk_write(bulk_updates, ordered=False)
                logger.info("Bulk update completed: Matched: %d, Modified: %d", result.matched_count, result.modified_count)
            except pymongo.errors.PyMongoError as e:
                logger.exception("Bulk update failed: %s", e)
                raise
        else:
            logger.info("No updates to process.")

    def _test_connection(self) -> None:
        try:
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful.")
        except Exception as e:
            logger.exception("Connection to MongoDB failed: %s", e)
            raise

    def _get_next_key(self, mongo_element: Dict[str, Any]) -> str:
        if 'detections_history' not in mongo_element:
            return 'v0'
        existing_keys = [int(s.replace('v', '')) for s in mongo_element['detections_history'].keys()]
        next_key = f'v{max(existing_keys) + 1}'
        return next_key

    def _update_rotation_query(
        self,
        current_mongo_element: Dict[str, Any],
        new_rotation: float,
        technique_used: str,
        confidence: float,
        is_new_best: bool
    ) -> UpdateOne:

        update_query = {}
        element_id = current_mongo_element['_id']
        next_key = self._get_next_key(current_mongo_element)

        if 'rotation' not in current_mongo_element.get('best_metadata', {}):
            is_new_best = True

        if is_new_best:
            update_query["best_metadata.rotation"] = next_key

        new_entry = {
            'rotation': new_rotation,
            'source': 'root_file',
            'technique_used': technique_used,
            'confidence': confidence,
            'usable_for_training': False
        }
        update_query[f"detections_history.{next_key}"] = new_entry

        return UpdateOne({"_id": ObjectId(element_id)}, {"$set": update_query})

    def _update_corners_query(
        self,
        current_mongo_element: Dict[str, Any],
        new_corners: List[float],
        technique_used: str,
        rotation_used: str,
        is_new_best: bool
    ) -> UpdateOne:
        update_query = {}
        element_id = current_mongo_element['_id']
        next_key = self._get_next_key(current_mongo_element)

        if 'corners' not in current_mongo_element.get('best_metadata', {}):
            is_new_best = True

        if is_new_best:
            update_query["best_metadata.corners"] = next_key

        new_entry = {
            'corners': new_corners,
            'source': f'root_file + rotate_{rotation_used}',
            'technique_used': technique_used,
            'usable_for_training': False
        }
        update_query[f"detections_history.{next_key}"] = new_entry

        return UpdateOne({"_id": ObjectId(element_id)}, {"$set": update_query})
