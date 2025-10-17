"""Create an end-to-end Levanter training animation for the main branch pipeline.

The animation is intentionally high-level and focuses on the conceptual stages
of a standard Levanter run: ingesting sharded text, tokenizing and packing, data
loading, forward/backward passes, optimizer updates, logging, and checkpointing.

It runs completely standalone (only Pillow is required) so it can be executed
outside the repository to explain the workflow to collaborators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Row:
    label: str
    tokens: List[str]
    colors: List[str]


@dataclass(frozen=True)
class Frame:
    title: str
    subtitle: str
    rows: List[Row]


@dataclass(frozen=True)
class AnimationConfig:
    columns: int = 10
    token_box_width: int = 95
    token_box_height: int = 56
    label_width: int = 180
    margin: int = 28
    frame_duration_ms: int = 2300
    background_color: str = "white"
    font_name: Optional[str] = None
    palette: Optional[dict[str, str]] = None


def _load_font(config: AnimationConfig, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if config.font_name:
        try:
            return ImageFont.truetype(config.font_name, size)
        except OSError:
            pass
    for candidate in ("Arial.ttf", "Helvetica.ttc", "SFNS.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pad(tokens: List[str], colors: List[str], config: AnimationConfig) -> tuple[List[str], List[str]]:
    padded_tokens = tokens[: config.columns]
    padded_colors = colors[: config.columns]
    while len(padded_tokens) < config.columns:
        padded_tokens.append("")
        padded_colors.append("#efefef")
    if len(padded_colors) < config.columns:
        padded_colors.extend(["#efefef"] * (config.columns - len(padded_colors)))
    return padded_tokens, padded_colors


def _draw_frame(frame: Frame, config: AnimationConfig) -> Image.Image:
    palette = config.palette or {
        "dataset": "#c6e2ff",
        "tokenizer": "#d0f4c1",
        "packing": "#ffe4ba",
        "batch": "#f8d0ff",
        "forward": "#ffd6e8",
        "loss": "#b3e5ff",
        "grad": "#d7bdfd",
        "update": "#d2f5ff",
        "log": "#f9ed8a",
        "checkpoint": "#d9d9d9",
        "background": "#efefef",
    }

    margin = config.margin
    width = (
        margin * 2
        + config.label_width
        + config.columns * config.token_box_width
    )
    row_block_height = config.token_box_height + margin // 2
    title_height = 70
    subtitle_height = 110
    height = margin * 2 + title_height + subtitle_height + len(frame.rows) * row_block_height

    image = Image.new("RGB", (width, height), config.background_color)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(config, 28)
    subtitle_font = _load_font(config, 20)
    label_font = _load_font(config, 18)
    token_font = _load_font(config, 16)

    draw.text((margin, margin), frame.title, fill="black", font=title_font)
    subtitle_y = margin + 42
    subtitle_lines = _wrap_text(draw, frame.subtitle, width - 2 * margin, subtitle_font)
    for line in subtitle_lines:
        draw.text((margin, subtitle_y), line, fill="black", font=subtitle_font)
        subtitle_y += 26

    grid_top = margin + title_height + subtitle_height

    for row_index, row in enumerate(frame.rows):
        tokens, colors = _pad(row.tokens, row.colors, config)
        top = grid_top + row_index * row_block_height
        label_box = [margin, top, margin + config.label_width - 12, top + config.token_box_height]
        draw.rounded_rectangle(label_box, radius=10, fill="#ffffff", outline="#cccccc", width=2)
        label_bbox = draw.textbbox((0, 0), row.label, font=label_font)
        label_x = label_box[0] + (config.label_width - 12 - (label_bbox[2] - label_bbox[0])) / 2
        label_y = top + (config.token_box_height - (label_bbox[3] - label_bbox[1])) / 2 - 2
        draw.text((label_x, label_y), row.label, fill="black", font=label_font)

        for col, (token, color) in enumerate(zip(tokens, colors)):
            x0 = margin + config.label_width + col * config.token_box_width
            y0 = top
            x1 = x0 + config.token_box_width - 10
            y1 = y0 + config.token_box_height
            draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=color or palette["background"], outline="black", width=2)
            if token:
                bbox = draw.textbbox((0, 0), token, font=token_font)
                text_x = x0 + (config.token_box_width - 10 - (bbox[2] - bbox[0])) / 2
                text_y = y0 + (config.token_box_height - (bbox[3] - bbox[1])) / 2 - 2
                draw.text((text_x, text_y), token, fill="black", font=token_font)

    return image


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def render_animation(frames: List[Frame], config: AnimationConfig, out_path: Path) -> None:
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


def build_frames(config: AnimationConfig) -> List[Frame]:
    palette = config.palette or {
        "dataset": "#c6e2ff",
        "tokenizer": "#d0f4c1",
        "packing": "#ffe4ba",
        "batch": "#f8d0ff",
        "forward": "#ffd6e8",
        "loss": "#b3e5ff",
        "grad": "#d7bdfd",
        "update": "#d2f5ff",
        "log": "#f9ed8a",
        "checkpoint": "#d9d9d9",
        "background": "#efefef",
    }

    cols = config.columns

    def fill(color: str) -> List[str]:
        return [color] * cols

    frames: List[Frame] = []

    frames.append(
        Frame(
            title="Step 1/8 · Sharded dataset ingestion",
            subtitle="Levanter streams text shards (e.g., Pile, SlimPajama) from cloud/object storage using AsyncDataset abstractions.",
            rows=[
                Row("Object store", [f"shard-{i:03d}" for i in range(cols)], fill(palette["dataset"])),
                Row("Tokenizer queue", ["pending"] * cols, [palette["background"]] * cols),
            ],
        )
    )

    token_ids = [f"{1000 + i}" for i in range(cols)]
    frames.append(
        Frame(
            title="Step 2/8 · Tokenization & normalization",
            subtitle="Text is lowercased, normalized, and converted to token IDs via the configured tokenizer (SentencePiece/BPE).",
            rows=[
                Row("Raw text", ["docA", "docA", "docB", "docB", "docC", "docC", "docD", "docD", "docE", "docE"], [palette["dataset"]] * cols),
                Row("Token IDs", token_ids, [palette["tokenizer"]] * cols),
            ],
        )
    )

    frames.append(
        Frame(
            title="Step 3/8 · Packing & UL2R corruption",
            subtitle="Token streams are packed into sequence windows; UL2R recipes apply span corruption and sentinel insertion before batching.",
            rows=[
                Row("Packed window", [f"seq{i}" for i in range(cols)], [palette["packing"]] * cols),
                Row("Sentinels", ["<extra_id_0>", "<extra_id_0>", "w17", "w18", "<extra_id_1>", "…", "<extra_id_1>", "w24", "w25", "…"], [palette["packing"]] * cols),
            ],
        )
    )

    frames.append(
        Frame(
            title="Step 4/8 · Distributed dataloading",
            subtitle="Async dataloaders shard batches per host/device, prefetching onto TPU/GPU memory while respecting PRNG keys for determinism.",
            rows=[
                Row("Global batch", [f"batch{i}" for i in range(cols)], [palette["batch"]] * cols),
                Row("Device slices", ["d0", "d0", "d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"], [palette["batch"]] * cols),
            ],
        )
    )

    frames.append(
        Frame(
            title="Step 5/8 · Forward pass",
            subtitle="Model layers (attention, MLP, normalization) run under JAX jit/pjit; activations stream through with named axes for clarity.",
            rows=[
                Row("Embedding →", ["tok", "tok", "tok"] + [""] * (cols - 3), [palette["forward"]] * cols),
                Row("Attention", ["QKᵀ", "softmax", "V", "resid", "MLP", "dropout", "resid", "norm", "…", "logits"], [palette["forward"]] * cols),
            ],
        )
    )

    frames.append(
        Frame(
            title="Step 6/8 · Loss & gradient computation",
            subtitle="Cross-entropy loss is aggregated with masking; JAX autodiff computes gradients via reverse-mode AD and gradient accumulation when configured.",
            rows=[
                Row("Loss mask", ["✓", "✓", "✓", "✓", "-", "-", "✓", "✓", "✓", "✓"], [palette["loss"]] * cols),
                Row("Gradients", ["∂W₁", "∂W₂", "∂b", "∂A", "∂V", "∂Norm", "∂Proj", "∂MLP", "…", "∂Emb"], [palette["grad"]] * cols),
            ],
        )
    )

    frames.append(
        Frame(
            title="Step 7/8 · Optimizer & EMA updates",
            subtitle="Optimizers (Adam/Muon/etc.) apply updates with distributed state sync; optional EMA/model averaging steps run post-update.",
            rows=[
                Row("Optimizer", ["adam m", "adam v", "lr", "clip", "μ", "ε", "EMA", "μ-avg", "weight", "bias"], [palette["update"]] * cols),
                Row("Params", ["W₁'", "W₂'", "b'", "A'", "V'", "Norm'", "Proj'", "MLP'", "…", "Emb'"] , [palette["update"]] * cols),
            ],
        )
    )

    frames.append(
        Frame(
            title="Step 8/8 · Logging & checkpointing",
            subtitle="Trackers publish metrics to WandB/JSON/TensorBoard; checkpoints stream to TensorStore or GCS with async uploads.",
            rows=[
                Row("Metrics", ["loss", "perplex", "lr", "grad_norm", "tokens", "throughput", "mem", "…", "", ""], [palette["log"]] * cols),
                Row("Checkpoint", ["step=42", "weights", "optimizer", "rng", "tracker", "tensorstore", "shard", "upload", "done", ""], [palette["checkpoint"]] * cols),
            ],
        )
    )

    return frames


def main() -> None:
    config = AnimationConfig()
    frames = build_frames(config)
    out_path = Path(__file__).with_name("levanter_training_pipeline.gif")
    render_animation(frames, config, out_path)
    print(f"Animation saved to {out_path}")


if __name__ == "__main__":
    main()
