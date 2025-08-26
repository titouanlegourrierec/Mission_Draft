"""
image_processing package

This package provides tools for batch processing, segmentation, and splitting of plant images, including:
        - BatchImageProcessor: for batch leaf splitting
        - LeafSplitter: for splitting leaves from images
        - LeafSegmenter: for segmenting leaves using Ilastik models
"""

from .batch_processor import BatchImageProcessor, split_leaves
from .leaf_segmenter import LeafSegmenter, segment_leaves
from .leaf_splitter import LeafSplitter

__all__ = [
    "BatchImageProcessor",
    "LeafSplitter",
    "LeafSegmenter",
    "split_leaves",
    "segment_leaves",
]
