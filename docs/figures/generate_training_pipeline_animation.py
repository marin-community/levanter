"""Generate a step-by-step animation of the Levanter training pipeline.

This script renders a GIF (or optionally an MP4) that walks through the
primary phases of training a language model with Levanter. Each frame focuses
on one stage while keeping the previous context visible, helping viewers build
an intuition for how configuration loading, data preparation, model
construction, optimization, and logging interplay during training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import textwrap

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text


@dataclass(frozen=True)
class StoryboardStep:
    """Container describing a single panel in the animation."""

    title: str
    summary: str
    details: Sequence[str]


STORYBOARD: Sequence[StoryboardStep] = (
    StoryboardStep(
        "Load Configuration",
        "Resolve experiment settings and seed the run for deterministic execution.",
        (
            "Parse base YAML/JSON configs with draccus.load.",
            "Apply CLI or sweep overrides and materialise experiment IDs.",
            "Seed PRNG streams and initialise tracker metadata.",
        ),
    ),
    StoryboardStep(
        "Assemble Mesh Resources",
        "Shape the device mesh and name axes used throughout the job.",
        (
            "Probe TPU/GPU topology to decide mesh axis sizes.",
            "Instantiate Haliax Axis objects for batch/model/heads dimensions.",
            "Create the PJIT mesh and register it with axis registries.",
        ),
    ),
    StoryboardStep(
        "Prepare Dataset Stream",
        "Build asynchronous pipelines that feed tokenised batches to devices.",
        (
            "Expand dataset configs into shard manifests (GCS/S3/local).",
            "Tokenise text, pack sequences, and apply map/shuffle transforms.",
            "Prefetch batches and stage host-device queues for overlap.",
        ),
    ),
    StoryboardStep(
        "Build Model & Optimizer",
        "Create Equinox modules and optimiser state aligned with named axes.",
        (
            "Initialise embeddings and Transformer blocks with split PRNG keys.",
            "Shape parameter PyTrees according to axis metadata.",
            "Configure optimiser hyperparameters and learning-rate schedule.",
        ),
    ),
    StoryboardStep(
        "Distribute State",
        "Shard parameters and runtime state across the device mesh.",
        (
            "Use hax.shard to partition parameters and optimiser slots.",
            "Broadcast weights with jax.device_put and replicate master copies.",
            "Scatter per-device PRNGs and gradient-accumulation buffers.",
        ),
    ),
    StoryboardStep(
        "Fetch Next Batch",
        "Stream preprocessed batches into device memory just-in-time.",
        (
            "Pull sequences from the async iterator and pad for mesh alignment.",
            "Assemble attention masks and loss masks on the host.",
            "Transfer data with double buffering to overlap compute and I/O.",
        ),
    ),
    StoryboardStep(
        "Forward Pass",
        "Run Transformer layers under jax.jit while respecting named axes.",
        (
            "Embed tokens and add positional encodings or rotary phases.",
            "Apply stacked attention/MLP blocks with activation checkpointing.",
            "Project hidden states through the LM head to produce logits.",
        ),
    ),
    StoryboardStep(
        "Compute Loss",
        "Derive training objectives and metrics from model outputs.",
        (
            "Align logits/targets, mask padding, and compute cross-entropy.",
            "Aggregate perplexity and accuracy across mesh axes with hax.mean.",
            "Record auxiliary stats (KL, regularisers) for logging.",
        ),
    ),
    StoryboardStep(
        "Backward Pass",
        "Differentiate the loss and prepare gradients for the optimiser.",
        (
            "Call jax.value_and_grad or eqx.filter_value_and_grad to obtain grads.",
            "Clip gradients, apply weight decay prep, and handle grad scaling.",
            "Collect gradient norms and microbatch statistics for trackers.",
        ),
    ),
    StoryboardStep(
        "Optimizer Update",
        "Update model parameters and scheduler state on each step.",
        (
            "Refresh optimiser slots (momentum, variance, Adafactor statistics).",
            "Apply parameter deltas respecting partition specs and weight decay.",
            "Advance learning-rate schedules and reset accumulation buffers.",
        ),
    ),
    StoryboardStep(
        "Checkpoint & Log",
        "Persist model state and surface metrics to monitoring hooks.",
        (
            "Stream metrics to WandB/TensorBoard/JSONL trackers.",
            "Trigger evaluation passes on validation datasets when scheduled.",
            "Write async checkpoints with parameters, optimiser state, and metadata.",
        ),
    ),
    StoryboardStep(
        "Loop Control",
        "Decide whether to continue training or exit gracefully.",
        (
            "Increment global step counters and update wall-clock accounting.",
            "Check stop criteria: max steps, tokens, time, or convergence triggers.",
            "Schedule the next iteration or begin shutdown/cleanup routines.",
        ),
    ),
)


class StoryboardView:
    """Encapsulates the Matplotlib artists needed to display the storyboard."""

    _BOX_COLORS = {
        "inactive": "#121212",
        "past": "#1f2933",
        "active": "#3a86ff",
    }
    _TEXT_COLORS = {
        "inactive": "#8a9ba8",
        "active": "#ffffff",
    }

    def __init__(self, storyboard: Sequence[StoryboardStep]) -> None:
        self.storyboard = storyboard
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.ax.set_axis_off()
        self.ax.set_facecolor("#0b0d12")

        self._boxes: List[FancyBboxPatch] = []
        self._titles: List[Text] = []
        self._descriptions: List[Text] = []
        self._numbers: List[Text] = []

        self._progress_bar = self.ax.barh(
            y=-0.8,
            width=0.0,
            height=0.3,
            left=0.5,
            color="#3a86ff",
        )[0]
        self._progress_bg = self.ax.barh(
            y=-0.8,
            width=8.0,
            height=0.3,
            left=0.5,
            color="#1f2933",
            alpha=0.4,
        )[0]

        self._create_layout()

    # Public API -----------------------------------------------------------

    @property
    def figure(self) -> plt.Figure:  # type: ignore[name-defined]
        return self.fig

    def artists(self) -> Iterable[Text | FancyBboxPatch]:
        return [
            *self._boxes,
            *self._titles,
            *self._descriptions,
            *self._numbers,
            self._progress_bar,
            self._progress_bg,
        ]

    def update(self, step_index: int, progress: float) -> None:
        total_steps = len(self.storyboard)

        progress = max(0.0, min(progress, 1.0))
        total_width = self._progress_bg.get_width()
        self._progress_bar.set_width(total_width * ((step_index + progress) / total_steps))

        for idx, (box, title, desc, number) in enumerate(
            zip(self._boxes, self._titles, self._descriptions, self._numbers)
        ):
            if idx < step_index:
                box.set_facecolor(self._BOX_COLORS["past"])
                box.set_alpha(0.8)
                title.set_color(self._TEXT_COLORS["inactive"])
                desc.set_color(self._TEXT_COLORS["inactive"])
                number.set_color(self._TEXT_COLORS["inactive"])
            elif idx == step_index:
                box.set_facecolor(self._BOX_COLORS["active"])
                box.set_alpha(1.0)
                title.set_color(self._TEXT_COLORS["active"])
                desc.set_color(self._TEXT_COLORS["active"])
                number.set_color(self._TEXT_COLORS["active"])
            else:
                box.set_facecolor(self._BOX_COLORS["inactive"])
                box.set_alpha(0.6)
                title.set_color(self._TEXT_COLORS["inactive"])
                desc.set_color(self._TEXT_COLORS["inactive"])
                number.set_color(self._TEXT_COLORS["inactive"])

    # Internal helpers -----------------------------------------------------

    def _create_layout(self) -> None:
        box_height = 0.85
        gap = 0.35
        y_positions = [idx * (box_height + gap) for idx in range(len(self.storyboard))]
        max_y = y_positions[-1] + box_height + gap

        self.ax.set_ylim(max_y + 0.2, -1.2)
        self.ax.set_xlim(0.0, 9.0)

        for idx, (step, y) in enumerate(zip(self.storyboard, y_positions)):
            box = FancyBboxPatch(
                (0.5, y),
                width=8.0,
                height=box_height,
                boxstyle="round,pad=0.3",
                linewidth=0.0,
                facecolor=self._BOX_COLORS["inactive"],
                alpha=0.6,
            )
            self.ax.add_patch(box)
            self._boxes.append(box)

            number = self.ax.text(
                0.8,
                y + box_height / 2,
                f"{idx + 1:02d}",
                va="center",
                ha="left",
                fontsize=18,
                fontweight="bold",
                color=self._TEXT_COLORS["inactive"],
            )
            self._numbers.append(number)

            title = self.ax.text(
                1.6,
                y + box_height / 2 + 0.18,
                step.title,
                va="center",
                ha="left",
                fontsize=16,
                fontweight="bold",
                color=self._TEXT_COLORS["inactive"],
            )
            self._titles.append(title)

            desc = self.ax.text(
                1.6,
                y + box_height / 2 - 0.24,
                _format_description(step, width=60),
                va="center",
                ha="left",
                fontsize=12,
                color=self._TEXT_COLORS["inactive"],
                wrap=True,
            )
            self._descriptions.append(desc)


def _format_description(step: StoryboardStep, width: int) -> str:
    lines: List[str] = []
    lines.append(
        textwrap.fill(
            step.summary,
            width=width,
            initial_indent="",
            subsequent_indent="",
        )
    )

    for detail in step.details:
        lines.append(
            textwrap.fill(
                detail,
                width=width,
                initial_indent="• ",
                subsequent_indent="  ",
            )
        )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/training_pipeline_animation.gif"),
        help="Destination file for the animation (suffix determines format).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the rendered animation.",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=10,
        help="Number of frames to hold each storyboard step.",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Disable looping when viewing the animation interactively.",
    )
    return parser.parse_args()


def _build_animation(view: StoryboardView, hold_frames: int, repeat: bool) -> FuncAnimation:
    total_steps = len(STORYBOARD)
    total_frames = total_steps * hold_frames

    def _update(frame: int) -> Iterable[Text | FancyBboxPatch]:
        step_index = min(frame // hold_frames, total_steps - 1)
        progress = (frame % hold_frames) / max(hold_frames - 1, 1)
        view.update(step_index, progress)
        return view.artists()

    return FuncAnimation(
        view.figure,
        _update,
        frames=total_frames,
        interval=1000 / 4,
        blit=False,
        repeat=repeat,
    )


def _choose_writer(output: Path, fps: int):
    suffix = output.suffix.lower()
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    if suffix == ".mp4":
        try:
            from matplotlib.animation import FFMpegWriter
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise SystemExit("FFMpegWriter unavailable; install ffmpeg to export MP4") from exc
        return FFMpegWriter(fps=fps)
    raise SystemExit(f"Unsupported output format '{suffix}'. Use .gif or .mp4.")


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    storyboard_view = StoryboardView(STORYBOARD)
    animation = _build_animation(storyboard_view, hold_frames=args.hold_frames, repeat=not args.no_repeat)

    writer = _choose_writer(args.output, fps=args.fps)
    animation.save(args.output, writer=writer)
    plt.close(storyboard_view.figure)
    print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    main()
