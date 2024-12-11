from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class _Image:
    TARGET_ASPECT_RATIOS = {
        "1:1": (1092, 1092),
        "3:4": (951, 1268),
        "2:3": (896, 1344),
        "9:16": (819, 1456),
        "1:2": (784, 1568),
    }

    def __init__(self, path):
        self.path = path
        self.image = Image.open(path)
        self.size = self.image.size
        self.format = self.image.format


    def rotate(self, rotation):
        self.image = self.image.rotate(-rotation, expand=True)
        self.size = self.image.size
        return self

    def crop(self, corners):
        width, height = self.size
        x1 = int(corners['x1'] * width)
        y1 = int(corners['y1'] * height)
        x2 = int(corners['x2'] * width)
        y2 = int(corners['y2'] * height)
        x3 = int(corners['x3'] * width)
        y3 = int(corners['y3'] * height)
        x4 = int(corners['x4'] * width)
        y4 = int(corners['y4'] * height)

        crop_xmin = min(x1, x2, x3, x4)
        crop_ymin = min(y1, y2, y3, y4)
        crop_xmax = max(x1, x2, x3, x4)
        crop_ymax = max(y1, y2, y3, y4)

        self.image = self.image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        self.size = self.image.size
        return self

    def save(self, output_path):
        self.image.save(output_path)

    def get_image(self):
        return self.image

    def get_path(self):
        return self.path

    def show(self):
        plt.imshow(self.image)
        plt.axis("off")
        plt.show()
    
    def get_type(self):
        if self.format is None:
            raise ValueError("Image format is not set. Unable to determine MIME type.")
        return f"image/{self.format.lower()}"

    def get_base64(self):
        buffer = BytesIO()
        self.image.save(buffer, format=self.format)
        buffer.seek(0)
        return base64.standard_b64encode(buffer.read()).decode("utf-8")

    def _calculate_aspect_ratio(self):
        return self.size[0] / self.size[1]

    def _find_closest_aspect_ratio(self):
        original_ratio = self._calculate_aspect_ratio()

        closest_ratio = None
        closest_dimensions = None
        closest_difference = float('inf')

        for ratio, dimensions in self.TARGET_ASPECT_RATIOS.items():
            target_width, target_height = dimensions
            target_ratio = target_width / target_height
            difference = abs(original_ratio - target_ratio)

            if difference < closest_difference:
                closest_difference = difference
                closest_ratio = ratio
                closest_dimensions = dimensions

        return closest_ratio, closest_dimensions

    def resize_aspect_ratio(self):
        _, target_dimensions = self._find_closest_aspect_ratio()
        target_width, target_height = target_dimensions

        self.image.thumbnail((target_width, target_height))
        self.size = self.image.size
        return self