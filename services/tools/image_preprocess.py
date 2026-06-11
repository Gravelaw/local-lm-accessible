from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AugmentationConfig:
    blur: bool = False
    skew: bool = False
    low_contrast: bool = False
    shadows: bool = False
    folds: bool = False
    mobile_camera_angle: bool = False
    partial_crop: bool = False
    seed: int = 0


def normalize_image_bytes(image_bytes: bytes) -> bytes:
    if not image_bytes:
        raise ValueError("image bytes are required")
    return image_bytes


def apply_augmentations(image: Image.Image, config: AugmentationConfig) -> Image.Image:
    rng = random.Random(config.seed)
    output = image.convert("RGB")
    if config.low_contrast:
        output = ImageEnhance.Contrast(output).enhance(0.55)
        output = ImageEnhance.Brightness(output).enhance(0.92)
    if config.shadows:
        output = _add_shadow(output, rng)
    if config.folds:
        output = _add_folds(output, rng)
    if config.skew:
        output = _skew(output, rng, max_degrees=3.0)
    if config.mobile_camera_angle:
        output = _perspective(output, rng)
    if config.partial_crop:
        output = _partial_crop(output, rng)
    if config.blur:
        output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.4)))
    return output


def _add_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.array(image).astype(np.float32)
    height, width = array.shape[:2]
    x_gradient = np.linspace(rng.uniform(0.72, 0.9), 1.0, width, dtype=np.float32)
    y_gradient = np.linspace(rng.uniform(0.82, 0.95), 1.0, height, dtype=np.float32)
    mask = np.minimum.outer(y_gradient, x_gradient)
    array *= mask[:, :, None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _add_folds(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.array(image).astype(np.float32)
    height, width = array.shape[:2]
    for _ in range(rng.randint(1, 3)):
        x = rng.randint(width // 5, max(width // 5 + 1, width - width // 5))
        fold_width = rng.randint(4, 12)
        start = max(0, x - fold_width)
        end = min(width, x + fold_width)
        array[:, start:end, :] *= rng.uniform(0.76, 0.88)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _skew(image: Image.Image, rng: random.Random, max_degrees: float) -> Image.Image:
    angle = rng.uniform(-max_degrees, max_degrees)
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")


def _perspective(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.array(image)
    height, width = array.shape[:2]
    offset = int(min(width, height) * rng.uniform(0.015, 0.045))
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32(
        [
            [rng.randint(0, offset), rng.randint(0, offset)],
            [width - rng.randint(0, offset), rng.randint(0, offset)],
            [width - rng.randint(0, offset), height - rng.randint(0, offset)],
            [rng.randint(0, offset), height - rng.randint(0, offset)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        array,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(warped)


def _partial_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    crop_x = rng.randint(0, max(1, width // 30))
    crop_y = rng.randint(0, max(1, height // 30))
    cropped = image.crop((crop_x, crop_y, width, height))
    output = Image.new("RGB", (width, height), "white")
    output.paste(cropped, (0, 0))
    return output
