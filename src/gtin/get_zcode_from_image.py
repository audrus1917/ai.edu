#!/usr/bin/env python3

"""The codes detection by `Zxing-cpp`."""

import os
import sys
import json
import argparse
import tkinter as tk

from pathlib import Path
from gettext import gettext as _

import cv2
import zxingcpp

from src.utils import init_logger

logger = init_logger(__name__)


def get_screen_size() -> tuple[int, int]:
    """Return the current screen size in pixels."""

    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def show_scaled_image(window_name: str, image) -> None:
    """Show an image scaled to fit the current screen."""

    screen_width, screen_height = get_screen_size()
    max_width = max(screen_width - 80, 1)
    max_height = max(screen_height - 120, 1)

    image_height, image_width = image.shape[:2]
    scale = min(max_width / image_width, max_height / image_height, 1.0)
    preview = image
    if scale < 1.0:
        preview = cv2.resize(
            image,
            (int(image_width * scale), int(image_height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(input_file: Path, output_path: Path) -> None:
    """Process a single image for barcode detection."""

    image = cv2.imread(str(input_file))
    if image is None:
        logger.error(_("Failed to load image: %(input)s"), {'input': input_file})
        return

    results = zxingcpp.read_barcodes(image)
    logger.info(
        _("Processed image: %(input)s, found %(count)d barcodes."), 
        {'input': input_file, 'count': len(results)}
    )
    if not results:
        show_scaled_image(str(input_file.name), image)

    input_stem = input_file.stem
    output_file = output_path / f"{input_stem}_results.json"
    if output_path:
        output_data = []
        for r in results:
            output_data.append({
                "format": str(r.format),
                "text": r.text
            })
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)


def main(input_path: Path, output_path: Path) -> None:
    """The main function for detecting codes in an image using Zxing-cpp."""

    if input_path.is_dir():
        logger.info(_("Processing images in folder: %(input)s"), {'input': input_path})     
        for file in input_path.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                process_image(file, output_path)
    else:
        process_image(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=_("The codes detection by `Zxing-cpp`.")
    )
    parser.add_argument(
        '--input',
        '-i',
        type=Path,
        required=True,
        help=_('The folder / file path for the input image(s) (QR, EAN, UPC, etc.).')
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        type=Path,
        help=_('The folder path for saving the barcode recognition results (QR, EAN, UPC, etc.).')
    )
    
    args = parser.parse_args()
    if not args.input.is_file() and not args.input.is_dir():
        logger.error(_("ERROR: The folder / file %(input)s not found!"), {'input': args.input})
        sys.exit(1)
    if not args.output.parent.exists():
        logger.error(_("ERROR: The folder %(output)s not found!"), {'output': args.output.parent})
        sys.exit(1)
    if not args.output.is_dir():
        logger.warning(_("WARNING: The folder %(input)s is not existed, create it"), {'input': args.input})
        os.makedirs(args.output, exist_ok=True)

    main(args.input, args.output)