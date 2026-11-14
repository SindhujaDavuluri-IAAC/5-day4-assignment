"""Classify a church photo using the published checkpoint or an offline model folder."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='Path to an exterior church photograph')
    parser.add_argument('--model-dir', type=Path, help='Offline folder containing model.pt and classes.json')
    args = parser.parse_args()
    if args.model_dir:
        weights_path = args.model_dir / 'model.pt'
        classes_path = args.model_dir / 'classes.json'
    else:
        from huggingface_hub import hf_hub_download
        repo = 'iaac-maai/day4_team5_church_style_classifier'
        weights_path = hf_hub_download(repo, 'model.pt')
        classes_path = hf_hub_download(repo, 'classes.json')

    with open(classes_path, encoding='utf-8') as stream:
        classes = json.load(stream)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()
    transform = models.ResNet18_Weights.DEFAULT.transforms()
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    with torch.inference_mode():
        scores = model(transform(image).unsqueeze(0)).softmax(1)[0]
    best = int(scores.argmax())
    print(json.dumps({
        'predicted_style': classes[best],
        'score': round(float(scores[best]), 4),
        'review_by_expert': float(scores[best]) < 0.6,
        'scores': {name: round(float(score), 4) for name, score in zip(classes, scores)},
    }, indent=2))


if __name__ == '__main__':
    main()
