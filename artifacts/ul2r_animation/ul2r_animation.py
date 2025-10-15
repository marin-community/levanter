"""Generate a UL2R training flow animation with minimal dependencies.

This script lives outside the main Levanter package tree so that it can be run
independently. It creates a GIF that narrates span corruption, sentinel
insertion, and PrefixLM packing for the UL2R R/X denoising objective.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import textwrap

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class UL2RConfig:
    """Control parameters for the toy UL2R demonstration."""

    length: int = 20
    mask_prob: float = 0.15
    mean_noise_span: float = 3.0
    random_roll: bool = True
    sentinels: tuple[str, ...] = tuple(f"<extra_id_{i}>" for i in range(8))
    pad_token: str = "<pad>"
    max_length: int = 28
    seed: int = 17
    frame_duration_ms: int = 2200
    background_color: str = "white"
    font_name: str | None = None
    token_box_width: int = 90
    token_box_height: int = 60
    palette: dict[str, str] = None


@dataclass(frozen=True)
class Frame:
    title: str
    subtitle: str
    tokens: List[str]
    colors: List[str]


def _random_segmentation(num_items: int, num_segments: int) -> List[int]:
    if num_segments <= 1:
        return [num_items]

    cut_points = sorted(random.sample(range(1, num_items), num_segments - 1))
    cuts = cut_points + [num_items]
    segments: List[int] = []
    prev = 0
    for cut in cuts:
        segments.append(cut - prev)
        prev = cut
    return segments


def random_spans_noise_mask(length: int, config: UL2RConfig) -> List[bool]:
    adjusted_length = max(length, 2)
    num_noise_tokens = int(round(adjusted_length * config.mask_prob))
    num_noise_tokens = max(1, min(num_noise_tokens, adjusted_length - 1))
    num_noise_spans = max(1, int(round(num_noise_tokens / config.mean_noise_span)))
    num_nonnoise_tokens = adjusted_length - num_noise_tokens

    noise_segments = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_segments = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    # Interleave non-noise and noise segments starting with non-noise span
    spans: List[bool] = []
    for nonnoise, noise in zip(nonnoise_segments, noise_segments):
        spans.extend([False] * nonnoise)
        spans.extend([True] * noise)

    mask = spans[:length]
    if len(mask) < length:
        mask.extend([False] * (length - len(mask)))

    if config.random_roll:
        offset = random.randint(0, adjusted_length - 1)
        mask = mask[offset:] + mask[:offset]

    return mask


def noise_span_to_unique_sentinel(tokens: List[str], mask: Iterable[bool], config: UL2RConfig) -> List[str]:
    output: List[str] = []
    sentinel_iter = iter(config.sentinels * ((len(tokens) // len(config.sentinels)) + 2))
    idx = 0
    mask_list = list(mask)
    while idx < len(tokens):
        if idx >= len(mask_list) or not mask_list[idx]:
            output.append(tokens[idx])
            idx += 1
            continue

        sentinel = next(sentinel_iter)
        output.append(sentinel)

        while idx < len(tokens) and mask_list[idx]:
            idx += 1

    return output


def roll_targets_to_suffix(targets: List[str], input_len: int, config: UL2RConfig) -> List[str]:
    capacity = config.max_length - input_len
    truncated = targets[:capacity]
    padding = [config.pad_token] * (config.max_length - input_len - len(truncated))
    return truncated + padding


def _pad_tokens(tokens: List[str], config: UL2RConfig) -> List[str]:
    if len(tokens) >= config.max_length:
        return tokens[: config.max_length]
    return tokens + [config.pad_token] * (config.max_length - len(tokens))


def build_frames(config: UL2RConfig) -> List[Frame]:
    random.seed(config.seed)
    tokens = [f"w{i}" for i in range(config.length)]
    mask = random_spans_noise_mask(config.length, config)
    inputs = noise_span_to_unique_sentinel(tokens, mask, config)
    targets = noise_span_to_unique_sentinel(tokens, [not m for m in mask], config)
    input_len = min(len(inputs), config.max_length)
    targets_suffix = roll_targets_to_suffix(targets, input_len, config)
    packed = inputs[:input_len] + targets_suffix

    palette = config.palette or {
        "base": "#c6e2ff",
        "noise": "#ffe4ba",
        "sentinel": "#d0f4c1",
        "target": "#e5d8ff",
        "packed_inputs": "#c6e2ff",
        "packed_targets": "#f8d0ff",
        "padding": "#ebebeb",
        "mask_row": "#ffd1a9",
        "loss_mask": "#b3e5ff",
    }

    padded_original = _pad_tokens(tokens, config)
    original_colors = [
        palette["base"] if i < len(tokens) else palette["padding"]
        for i in range(config.max_length)
    ]

    noise_highlight_colors = [
        palette["noise"] if i < len(mask) and mask[i] else original_colors[i]
        for i in range(config.max_length)
    ]

    mask_tokens = ["1" if m else "0" for m in mask]
    padded_mask_tokens = _pad_tokens(mask_tokens, config)
    mask_colors = [
        palette["mask_row"] if i < len(mask) and mask[i] else palette["padding"]
        for i in range(config.max_length)
    ]

    padded_inputs = _pad_tokens(inputs, config)
    input_colors = [
        palette["sentinel"] if padded_inputs[i].startswith("<extra_id_") else original_colors[i]
        if padded_inputs[i] != config.pad_token
        else palette["padding"]
        for i in range(config.max_length)
    ]

    padded_targets = _pad_tokens(targets, config)
    target_colors = [
        palette["sentinel"] if padded_targets[i].startswith("<extra_id_") else palette["target"]
        if padded_targets[i] != config.pad_token
        else palette["padding"]
        for i in range(config.max_length)
    ]

    padded_packed = _pad_tokens(packed, config)
    packed_colors = [
        palette["packed_inputs"]
        if i < input_len and padded_packed[i] != config.pad_token
        else palette["packed_targets"]
        if padded_packed[i] != config.pad_token
        else palette["padding"]
        for i in range(config.max_length)
    ]

    loss_tokens = [
        "✓" if (i >= input_len and padded_packed[i] != config.pad_token) else "-"
        for i in range(config.max_length)
    ]
    loss_colors = [
        palette["loss_mask"] if token == "✓" else palette["padding"]
        for token in loss_tokens
    ]

    frames: List[Frame] = [
        Frame(
            title="Step 1/7 · Original token stream",
            subtitle="Tokens w0…w19 form the packed input; grey cells are padding.",
            tokens=padded_original,
            colors=original_colors,
        ),
        Frame(
            title="Step 2/7 · Sampled noise spans",
            subtitle="Orange tokens belong to masked spans selected by UL2R noise sampling.",
            tokens=padded_original,
            colors=noise_highlight_colors,
        ),
        Frame(
            title="Step 3/7 · Binary mask (1 = noise span)",
            subtitle="Mask controls which tokens become sentinels (1) versus remain in the prefix (0).",
            tokens=padded_mask_tokens,
            colors=mask_colors,
        ),
        Frame(
            title="Step 4/7 · Inputs with sentinels",
            subtitle="Each masked span collapses to a unique sentinel token while intact tokens stay blue.",
            tokens=padded_inputs,
            colors=input_colors,
        ),
        Frame(
            title="Step 5/7 · Targets = removed spans",
            subtitle="Targets contain sentinels plus the original span contents in order of appearance.",
            tokens=padded_targets,
            colors=target_colors,
        ),
        Frame(
            title="Step 6/7 · Packed PrefixLM example",
            subtitle="Inputs (blue) are followed by targets (purple); model attends bidirectionally then causally.",
            tokens=padded_packed,
            colors=packed_colors,
        ),
        Frame(
            title="Step 7/7 · Loss mask (✓ = contributes)",
            subtitle="Loss is attached only to generated target tokens; inputs and padding are ignored.",
            tokens=loss_tokens,
            colors=loss_colors,
        ),
    ]

    return frames


def _load_font(config: UL2RConfig, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if config.font_name:
        try:
            return ImageFont.truetype(config.font_name, size)
        except OSError:
            pass
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_frame(frame: Frame, config: UL2RConfig) -> Image.Image:
    margin = 24
    width = margin * 2 + config.token_box_width * config.max_length
    subtitle_space = 90
    height = margin * 2 + config.token_box_height + subtitle_space + 40

    image = Image.new("RGB", (width, height), config.background_color)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(config, 26)
    subtitle_font = _load_font(config, 18)
    token_font = _load_font(config, 16)

    draw.text((margin, margin), frame.title, fill="black", font=title_font)
    title_bbox = draw.textbbox((margin, margin), frame.title, font=title_font)
    title_height = title_bbox[3] - title_bbox[1]

    subtitle_y = margin + title_height + 8
    wrapper = textwrap.TextWrapper(width=80)
    subtitle_line_height_bbox = draw.textbbox((0, 0), "Ag", font=subtitle_font)
    subtitle_line_height = subtitle_line_height_bbox[3] - subtitle_line_height_bbox[1]

    for line in wrapper.wrap(frame.subtitle):
        draw.text((margin, subtitle_y), line, fill="black", font=subtitle_font)
        subtitle_y += subtitle_line_height + 4

    grid_top = subtitle_y + 16

    for i, (token, color) in enumerate(zip(frame.tokens, frame.colors)):
        x0 = margin + i * config.token_box_width
        y0 = grid_top
        x1 = x0 + config.token_box_width - 8
        y1 = y0 + config.token_box_height
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=10,
            fill=color,
            outline="black",
            width=2,
        )

        if token:
            bbox = draw.textbbox((0, 0), token, font=token_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x0 + (config.token_box_width - 8 - text_width) / 2
            text_y = y0 + (config.token_box_height - text_height) / 2 - 2
            draw.text((text_x, text_y), token, fill="black", font=token_font)

    return image


def render_animation(frames: List[Frame], config: UL2RConfig, out_path: Path) -> None:
    images = [_draw_frame(frame, config) for frame in frames]
    durations = [config.frame_duration_ms] * len(images)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )


def main() -> None:
    config = UL2RConfig()
    frames = build_frames(config)
    out_path = Path(__file__).with_name("ul2r_training_flow.gif")
    render_animation(frames, config, out_path)
    print(f"Animation saved to {out_path}")


if __name__ == "__main__":
    main()
