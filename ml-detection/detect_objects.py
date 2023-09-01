import argparse
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, YolosForObjectDetection

# To filter out a specific warning
warnings.filterwarnings("ignore", category=FutureWarning)

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        help="path to image in local or url",
        default="https://raw.githubusercontent.com/SamuelHudec/object-detection/main/images/test_image.jpeg",
    )
    return parser.parse_args()


def save_results(pil_img, prob, boxes, model):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f"{model.config.id2label[cl.item()]}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"results/output_image_{timestamp}.png"
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.1, dpi=300)


def main(args: argparse.Namespace) -> None:
    if args.image_path.startswith("https://"):
        image = Image.open(requests.get(args.image_path, stream=True).raw)
    else:
        image = Image.open(args.image_path)

    feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small")
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7  # (0.7) future hyper-parameter

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]["boxes"]

    save_results(image, probas[keep], bboxes_scaled[keep], model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
