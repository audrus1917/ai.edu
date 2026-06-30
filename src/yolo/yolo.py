#!/usr/bin/env python3

"""The objects detection by `YOLO`."""

import sys
import argparse

from pathlib import Path
from gettext import gettext as _

import cv2

from ultralytics import YOLO

from src.utils import init_logger

logger = init_logger(__name__)


def main(input_path: Path, output_path: Path) -> None:
    """The main function for detecting objects in an image using YOLO."""

    model = YOLO(
        '_models/yolov8s-fashionpedia-1.onnx',
        task='detect'
    )

    # Processing the image with clothing and footwear
    results = model(input_path)

    # Drawing boxes around objects
    annotated_frame = results[0].plot()


    

    cv2.imwrite(output_path, annotated_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=_("Тест распознавания текста на фотографии этикетки товаров.")
    )
    parser.add_argument(
        '--input',
        '-i',
        type=Path,
        help=_('Путь к изображению для распознавания')
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=_("Выходной файл")
    )
    args = parser.parse_args()
    if not args.input.is_file():
        logger.error(_("Ошибка: Файл %(input)s не найден!"), {'input': args.input})
        sys.exit(1)

    main(args.input, args.output)