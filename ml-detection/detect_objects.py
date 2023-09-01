import torch
from PIL import Image
import requests
from transformers import AutoFeatureExtractor
from transformers import YolosForObjectDetection
import matplotlib.pyplot as plt
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        help="path to image in local or url",
        required=True,
        default="images/test_image.jpeg"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:





if __name__ == "__main__":
    args = parse_args()
    main(args)