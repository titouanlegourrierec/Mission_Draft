"""
This module provides the BatchImageProcessor class and a split_leaves function for batch processing of images.
It uses the LeafSplitter class to segment and save leaves from all images in a directory, and provides utilities
for validating directories and gathering output statistics.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

import src.config as config
from src.image_processing.leaf_splitter import LeafSplitter


def split_leaves(input_dir: str, output_dir: str, color_space: str = "RGB") -> None:
    """
    Split leaves in all images within the input directory and save them to the output directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory where split images will be saved.
        color_space (str): Color space to use for saving images ('RGB', 'YUV', 'HSV', 'LAB', 'HLS').
    """
    processor = BatchImageProcessor(input_dir, output_dir, color_space)
    is_valid, error_message = processor.validate_directories()
    if not is_valid:
        raise ValueError(f"Directory validation failed: {error_message}")

    processor.process_images()


class BatchImageProcessor:
    """
    A class to handle batch processing of images using LeafSplitter.
    color_space: str: Color space to use for saving images ('RGB', 'YUV', 'HSV', 'LAB', 'HLS').
    """

    def __init__(self, input_dir: str, output_dir: str, color_space: str = "RGB"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.color_space = color_space.upper()

    def validate_directories(self) -> Tuple[bool, str]:
        """
        Validate input and output directories.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            return False, "Input directory does not exist."
        if not self.output_dir.exists() or not self.output_dir.is_dir():
            return False, "Output directory does not exist."
        return True, ""

    def find_images(self) -> List[Path]:
        """
        Find all image files in the input directory.

        Returns:
            List[Path]: List of image file paths
        """
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
        images = [
            img_path
            for ext in image_extensions
            for img_path in self.input_dir.glob(ext)
            if img_path.is_file()
        ]
        return images

    def process_images(self, progress_callback=None) -> None:
        """
        Process all images in the input directory.

        Args:
            progress_callback: Optional callback function to report progress
        """
        images = self.find_images()

        for idx, img_path in enumerate(images, 1):
            splitter = LeafSplitter(
                str(img_path),
                str(self.output_dir),
                color_space=self.color_space,
            )
            splitter.split_and_save()
            if progress_callback:
                progress_callback(idx, len(images))

    def get_output_stats(
        self,
    ) -> Tuple[Optional[List[int]], Optional[List[int]], List[Path]]:
        """
        Compute statistics about the processed images.

        Returns:
            Tuple[Optional[List[int]], Optional[List[int]], List[Path]]:
            (widths, heights, image_paths)
        """
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
        output_images = [
            f
            for ext in image_extensions
            for f in self.output_dir.glob(ext)
            if f.is_file()
        ]

        widths, heights = [], []
        for img_path in output_images:
            try:
                with Image.open(img_path) as im:
                    # Convert pixels to millimeters: mm = (pixels / DPI) * 25.4
                    widths.append(im.width / config.DPI * 25.4)
                    heights.append(im.height / config.DPI * 25.4)
            except Exception:
                continue

        return widths, heights, output_images
