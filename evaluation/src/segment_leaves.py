"""
Command-line interface for leaf segmentation using a trained Ilastik model.
This script provides a convenient way to segment leaf images using the
leaf_segmenter module.
"""

import argparse
import logging
import os
import sys

# Add the parent directory to sys.path to allow importing the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from src.image_processing.leaf_segmenter import segment_leaves
except ImportError:
    print("Error: Could not import segment_leaves. Make sure the module exists.")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Segment leaf images using a trained Ilastik model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained Ilastik model file (.ilp)",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="./evaluation/images_to_segment",
        help="Path to the directory containing input images to segment (default: ./evaluation/images_to_segment)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation/segmented_images",
        help="Path to the directory where segmented images will be saved (default: ./evaluation/segmented_images)",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose logging output",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to parse arguments and run leaf segmentation.
    """
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate paths
    for path_name, path_value in [
        ("model_path", args.model_path),
        ("input_path", args.input_path),
    ]:
        if not os.path.exists(path_value):
            logging.error(f"Error: {path_name} '{path_value}' does not exist.")
            sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Check if model file has the right extension
    if not args.model_path.lower().endswith(".ilp"):
        logging.warning(
            f"Warning: model_path '{args.model_path}' does not have .ilp extension."
        )

    try:
        logging.info(f"Starting leaf segmentation with model: {args.model_path}")
        logging.info(f"Input directory: {args.input_path}")
        logging.info(f"Output directory: {args.output_path}")

        # Run segmentation
        segment_leaves(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
        )

        logging.info(
            f"Segmentation completed successfully. Results saved to {args.output_path}"
        )
        logging.info(
            f"A CSV report with pixel class statistics has been generated at {os.path.join(args.output_path, 'results.csv')}"
        )

    except Exception as e:
        logging.error(f"Error during segmentation: {str(e)}")
        if args.verbose:
            import traceback

            logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
