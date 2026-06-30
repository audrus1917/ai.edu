"""The image recognition example using YOLO11 nano model."""

import sys
import argparse
import logging

from pathlib import Path

import cv2

from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", type=str)
    args = parser.parse_args()

    image_file = Path(args.imagefile)
    if not image_file.is_file():
        logger.error(
            f"Файл {image_file} не найден. Пожалуйста, укажите"
            f" существующий файл изображения."
        )
        sys.exit(1) 

    image = Image.open(image_file)

    model = YOLO("yolo11n.pt")
    results = model(image, save=True)

    logger.debug(f"Results: {results}")

    logger.info(f"Результаты распознавания сохранены в папке 'runs/detect/'")

    logger.debug(dir(results))
    for idx, result in enumerate(results):
        logger
        boxes = result.boxes
        for box in boxes:
            # Получаем ID класса (например, 0 - person, 5 - bus)
            class_id = int(box.cls[0])
            # Получаем текстовое название класса
            class_name = model.names[class_id]
            # Получаем уверенность модели (confidence score)
            confidence = float(box.conf[0])
            logger.info(f"Обнаружен объект: {class_name} (Уверенность: {confidence:.2f})")
