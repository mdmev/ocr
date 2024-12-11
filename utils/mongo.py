import pymongo


class Mongo:
    def __init__(self, connection_uri):
        self.client = pymongo.MongoClient(connection_uri)

    def extract_metadata(self, image_path):
        image_name = image_path.split('/')[-1]
        document = self.client.data_repository.files.find_one({"blob_filename": image_name})
        if not document:
            raise ValueError(f"Image '{image_name}' not found in the database.")

        rotation_key = document['best_metadata']['rotation']
        rotation = document['detections_history'][rotation_key]['rotation']

        corners_key = document['best_metadata']['corners']
        corners = document['detections_history'][corners_key]['corners']

        return {"rotation": rotation, "corners": corners}
