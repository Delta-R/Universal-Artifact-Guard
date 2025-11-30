#!/usr/bin/env python3
"""
针对单张图片的 D3 模型推理脚本。
输入：图片路径 + 训练好的 checkpoint
输出：预测分数（越大越偏向 fake）以及基于指定阈值的标签判断。
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}

# ===== 固定配置：如需调整，直接修改这里 ===== #
CHECKPOINT_PATH = Path("path/to/save/checkpoints/train_d3/model_epoch_best.pth")
MODEL_NAME = "ViT-L/14"
GRANULARITY = 14
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORM = True  # 若需要排查，可改为 False
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对单张图片进行真假检测推理")
    parser.add_argument("image", type=Path, help="待测试的图片路径")
    return parser.parse_args()


def build_transform(arch: str, is_norm: bool) -> transforms.Compose:
    stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
    ops = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    if is_norm:
        ops.append(transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]))
    return transforms.Compose(ops)


def load_model(checkpoint: Path, model_name: str, granularity: int, device: torch.device):
    from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer

    model = CLIPModelShuffleAttentionPenultimateLayer(
        model_name,
        shuffle_times=1,
        original_times=1,
        patch_size=[granularity],
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.attention_head.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def infer_single_image(
    image_path: Path,
    model,
    transformer: transforms.Compose,
    device: torch.device,
) -> float:
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片：{image_path}")
    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件扩展名：{image_path.suffix}，仅支持 {sorted(ALLOWED_EXTENSIONS)}"
        )

    img = Image.open(image_path).convert("RGB")
    tensor = transformer(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        if output.shape[-1] == 2:
            output = output[:, 0]
        score = torch.sigmoid(output).item()
    return score


def format_result(score: float, threshold: float) -> Tuple[str, str]:
    label = "fake" if score > threshold else "real"
    confidence = f"{score:.4f}"
    return label, confidence


def main() -> None:
    args = parse_args()
    device = DEVICE

    transformer = build_transform(arch=MODEL_NAME, is_norm=USE_NORM)
    model = load_model(CHECKPOINT_PATH, MODEL_NAME, GRANULARITY, device)

    score = infer_single_image(args.image, model, transformer, device)
    label, confidence = format_result(score, THRESHOLD)

    print(f"fake_probability={confidence}")
    print(f"predicted_label={label}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
