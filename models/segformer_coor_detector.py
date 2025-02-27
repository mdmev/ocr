import numpy as np
import argparse
import os
import json
import shutil
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt  # For saving images

from alive_progress import alive_bar
from utils import get_coords
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


PATH = os.getcwd()

def save_mask(image, mask, filename):
    """Save the original image and its segmentation mask as separate files."""
    img_save_path = os.path.join(MASK_DIR, f"{filename}_original.jpg")
    mask_save_path = os.path.join(MASK_DIR, f"{filename}_mask.png")

    # Convert mask to a PIL image (grayscale)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Scale 0-1 mask to 0-255

    # Save images
    image.save(img_save_path)
    mask_image.save(mask_save_path)

    print(f"Saved debug images: {img_save_path}, {mask_save_path}")


def inference(model_path, images_folder, output_file, iterations, save_debug_masks=True, model_version):
    model = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/mit-{model_version}", num_labels=1)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth'), weights_only=True))

    processor = AutoImageProcessor.from_pretrained(f"nvidia/mit-{model_version}")

    device = "cuda"
    model.to(device)
    model.eval()

    path = os.path.join(PATH, images_folder)
    files = os.listdir(path)

    corners = []

    # Optionally enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        with alive_bar(len(files)) as bar:
            for file in files:
                try:
                    img_path = os.path.join(path, file)
                    img = Image.open(img_path).convert("RGB")
                    width, height = img.size  

                    inputs = processor(images=img, return_tensors="pt").to(device)

                    # Use mixed precision for faster inference
                    with torch.cuda.amp.autocast():
                        output = model(**inputs)
                        logits = output.logits
                        upsampled_logits = nn.functional.interpolate(
                            logits,
                            size=img.size[::-1],  # (height, width)
                            mode='bilinear',
                            align_corners=False
                        )
                        # Perform sigmoid and thresholding on GPU
                        seg = (torch.sigmoid(upsampled_logits) > 0.5).to(torch.uint8)
                    
                    # Optionally save debug mask (convert to CPU only when needed)
                    if save_debug_masks:
                        seg_np = seg.cpu().squeeze().numpy()
                        save_mask(img, seg_np, os.path.splitext(file)[0])
                    
                    # If get_coords can work with GPU tensors, pass seg directly
                    seg_tensor = seg.squeeze()  # adjust dimensions as needed

                    coords = get_coords(seg_tensor, iterations)
                    if coords is None:
                        print(f"get_coords returned None for image {file}. Skipping...")
                        continue

                    norm_factor = np.array([width, height], dtype=np.float32)
                    coords = coords.astype(np.float32) / norm_factor

                    coords_json = {
                        "filename": file,
                        "corners": [
                            float(coords[1][0]), float(coords[1][1]),
                            float(coords[0][0]), float(coords[0][1]),
                            float(coords[3][0]), float(coords[3][1]),
                            float(coords[2][0]), float(coords[2][1])
                        ]
                    }
                    corners.append(coords_json)

                except Exception as e:
                    print(f'Image {file} failed. Error: {e}')
                    shutil.copy(img_path, os.path.join(PATH, "bad_inferenceb0"))

                bar()

    with open(output_file + ".json", 'w') as f:
        json.dump(corners, f, indent=4)


def argparser():
    parser = argparse.ArgumentParser(
        description="Script for training text corner detection model"
    )
    parser.add_argument("--model", default='', type=str)
    parser.add_argument("--source_dir", default='', type=str)
    parser.add_argument("--output_dir", default='', type=str)
    parser.add_argument('--iterations', default=1000, type=int)
    paraser.add_argument("--model_version",default="b0",type=str)

    parser.add_argument(
        "--freeze", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    world_size = torch.cuda.device_count()
    
    print(f"Running on {world_size} GPUs")
    
    inference(
        model_path=args.model,
        images_folder=args.source_dir,
        output_file=args.output_dir,
        iterations=args.iterations,
        model_version=args.model_version
    )
