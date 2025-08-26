"""
Command-line interface for generating ground truth masks from image annotations.
This script takes image annotations in the form of a JSON file and generates ground truth masks
for each image based on the provided segmentation information.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Default background value for mask backgrounds (white).
# Change this value to set a different background color for your segmentation masks.
BACKGROUND_COLOR = 255


def load_color_map(colormap_path: str = "./src/color_map.json") -> Dict[str, int]:
    """
    Load the color map used for generating masks from a JSON file.

    Args:
        colormap_path (str): Path to the JSON file containing the color map.

    Returns:
        dict: Dictionary mapping class names to color values.
    """
    logging.info(f"Loading color map from {colormap_path}")
    try:
        with open(colormap_path, "r") as color_file:
            return json.load(color_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load color map: {e}")
        raise


def load_annotations(
    groundtruth_json_path: str = "./evaluation/ground_truth.json",
) -> Dict[str, Any]:
    """
    Load image annotations from a VIA (VGG Image Annotator) JSON file.

    Args:
        groundtruth_json_path (str): Path to the VIA annotation JSON file.

    Returns:
        dict: Dictionary containing image metadata and region annotations.
    """
    logging.info(f"Loading annotations from {groundtruth_json_path}")
    try:
        with open(groundtruth_json_path, "r") as annotation_file:
            data = json.load(annotation_file)
        return data["_via_img_metadata"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.error(f"Failed to load annotations: {e}")
        raise


def create_groundtruth(
    img_shape: Tuple[int, int, int],
    regions: List[Dict[str, Any]],
    color_map: Dict[str, int],
    background_color: int = BACKGROUND_COLOR,
) -> np.ndarray:
    """
    Create a groundtruth image from region annotations and a color map.

    Args:
        img_shape (tuple): Shape of the original image (height, width, channels).
        regions (list): List of region annotations for the image.
        color_map (dict): Dictionary mapping class names to color values.
        background_color (int): Value for the background color of masks (default: 255 for white).

    Returns:
        np.ndarray: The generated groundtruth image as a NumPy array.
    """
    height, width = img_shape[:2]
    mask = np.full((height, width), background_color, dtype=np.uint8)

    # Process each annotated region
    for region in regions:
        shape_attr = region["shape_attributes"]
        region_attr = region["region_attributes"]

        # Get and clean class name
        class_name = region_attr.get("name", "").replace(" ", "")

        # Skip regions with unknown class names
        if class_name not in color_map:
            logging.warning(f"Class '{class_name}' not in color_map, skipping region.")
            continue

        # Get color value for the class
        class_color = color_map[class_name]

        # Convert different shape types to polygon points
        shape_type = shape_attr["name"]
        if shape_type == "polygon":
            # Create points array from x, y coordinates
            points = np.array(
                list(zip(shape_attr["all_points_x"], shape_attr["all_points_y"]))
            )
        elif shape_type == "circle":
            # Convert circle to polygon points
            center = (int(shape_attr["cx"]), int(shape_attr["cy"]))
            radius = int(shape_attr["r"])
            points = np.array(cv2.ellipse2Poly(center, (radius, radius), 0, 0, 360, 1))
        else:
            logging.warning(f"Unsupported shape: {shape_type}, skipping region.")
            continue

        # Draw filled contour with class color
        cv2.drawContours(mask, [points], -1, class_color, -1)

    return mask


def generate_groundtruth(
    groundtruth_json_path: str = "./evaluation/ground_truth.json",
    images_to_segment_dir: str = "./evaluation/images_to_segment",
    groundtruth_dir: str = "./evaluation/groundtruth",
    colormap_path: str = "./src/color_map.json",
    background_color: int = BACKGROUND_COLOR,
):
    """
    Generate groundtruth images for all annotated images using the provided color map.

    Args:
        groundtruth_json_path (str): Path to the VIA annotation JSON file.
        images_to_segment_dir (str): Directory containing the original images.
        groundtruth_dir (str): Directory where the generated groundtruth images will be saved.
        colormap_path (str): Path to the JSON file containing the color map.
        background_color (int): Value for the background color of masks (default: 255 for white).

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(groundtruth_dir, exist_ok=True)

    # Load class color mapping and annotations
    color_map = load_color_map(colormap_path)
    annotations = load_annotations(groundtruth_json_path)

    logging.info(f"Generating groundtruth for {len(annotations)} images.")

    # Process each annotated image
    for image_metadata in annotations.values():
        filename = image_metadata["filename"]
        img_path = os.path.join(images_to_segment_dir, filename)

        # Load source image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            logging.error(f"Failed to load image: {img_path}")
            continue

        # Create groundtruth mask from annotations
        mask = create_groundtruth(
            image.shape, image_metadata["regions"], color_map, background_color
        )

        # Save generated mask
        output_path = os.path.join(groundtruth_dir, filename)
        cv2.imwrite(output_path, mask)

    logging.info(f"Groundtruth generation completed. Masks saved to {groundtruth_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Generate groundtruth masks from VIA annotation JSON."
    )

    parser.add_argument(
        "--images_to_segment_dir",
        type=str,
        default="./evaluation/images_to_segment",
        help="Directory containing source images (default: ./evaluation/images_to_segment)",
    )

    parser.add_argument(
        "--groundtruth_dir",
        type=str,
        default="./evaluation/ground_truth",
        help="Directory to save groundtruth mask images (default: ./evaluation/ground_truth)",
    )

    parser.add_argument(
        "--colormap_path",
        type=str,
        default="./src/color_map.json",
        help="Path to the class color map JSON file (default: ./src/color_map.json)",
    )

    parser.add_argument(
        "--groundtruth_json_path",
        type=str,
        default="./evaluation/ground_truth.json",
        help="Path to the annotation JSON file (default: ./evaluation/ground_truth.json)",
    )

    parser.add_argument(
        "--background_color",
        type=int,
        default=BACKGROUND_COLOR,
        help=f"Background color value for mask (0-255, default: {BACKGROUND_COLOR} for white)",
    )

    args = parser.parse_args()

    try:
        logging.info("Starting groundtruth generation.")
        generate_groundtruth(
            groundtruth_json_path=args.groundtruth_json_path,
            images_to_segment_dir=args.images_to_segment_dir,
            groundtruth_dir=args.groundtruth_dir,
            colormap_path=args.colormap_path,
            background_color=args.background_color,
        )
        logging.info("Groundtruth generation completed successfully.")
    except Exception as e:
        logging.error(f"Groundtruth generation failed: {e}")
        import traceback

        logging.error(traceback.format_exc())
        exit(1)
