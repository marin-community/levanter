# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Union
import traceback

import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax
from haliax.partitioning import ResourceMapping

import levanter
from levanter.books.util import create_pz_histogram, create_pz_histogram_linear
from levanter.callbacks import StepInfo
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfigBase
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.models.loss import next_token_loss
from levanter.utils.hf_utils import HfTokenizer
from levanter.tracker.histogram import Histogram


@dataclass
class PzInnerLoopConfig:
    datasets: Optional[List[str]] = None
    doc_tokens: Optional[int] = None
    chunk_size: int = 512
    prompt_tokens: Optional[int] = None
    cursor_inc_tokens: int = 1
    num_documents: int = 1
    mode: str = "sliding"  # one of: "sliding" (default), "first"
    eval_batch_size: Optional[int] = 64  # batch across docs in 'first' mode
    histogram: bool = False
    histogram_linear: bool = True
    pz_threshold: float = 1e-4
    pz_npz: bool = False
    # Only log histogram artifact when (global_step % histogram_every_steps == 0). If None, log whenever histogram=True
    histogram_every_steps: Optional[int] = None
    decode_preview: Optional[int] = None
    verify_treecache: bool = False
    # Verbose printing for debug; default false so configs need not set it
    verbose: bool = False


def pz_eval_callback(
    config: PzInnerLoopConfig,
    tokenizer: HfTokenizer,
    axis_resources: ResourceMapping,
    mp,
    data_config: Union[LMMixtureDatasetConfig, SingleDatasetLMConfigBase],
):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    decode_once_state: dict = {}
    # Persistent state across callback invocations (per process)
    caches_state = None  # set on first call
    selected_indices_by_ds: dict[str, List[int]] = {}

    def _ts():
        return time.strftime("%H:%M:%S", time.localtime())

    def _log(msg: str, *, all_hosts: bool = False):
        # Gate all console printing behind config.verbose (default False)
        if not config.verbose:
            return
        if all_hosts or jax.process_index() == 0:
            print(f"[PZ][{_ts()}][proc={jax.process_index()}] {msg}", flush=True)

    def _compute_logprob_for_tokens(model: LmHeadModel, tokens_1d: np.ndarray, prompt_len: int) -> float:
        N = int(tokens_1d.shape[0])
        Pos = model.Pos.resize(N)
        toks_named = hax.named(np.array(tokens_1d, dtype=np.int32), Pos)
        ex = LmExample.from_prompt_and_completion(Pos, toks_named, prompt_length=int(prompt_len), ignore_id=pad_id)
        m = model
        if mp is not None:
            m = mp.cast_to_compute(m)
        with hax.axis_mapping(axis_resources):
            logits = m(ex.tokens, attn_mask=ex.attn_mask)
            logits = logits.astype(jnp.float32)
            nll = next_token_loss(
                Pos=Pos, Vocab=m.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None
            )
            total_nll = hax.sum(nll, axis=Pos).array
        return -float(np.array(total_nll))

    def _run_for_model(model: LmHeadModel, *, log_histogram_now: bool, curr_step: int):
        # Only process 0 does tracker I/O. All hosts print phase timings for debugging.
        _log(f"step={curr_step} | BEGIN pz_eval inner-loop")
        _log(
            f"jax world: process_count={jax.process_count()} local_device_count={jax.local_device_count()} total_device_count={len(jax.devices())}",
            all_hosts=True,
        )

        nonlocal caches_state
        eval_start_time = time.time()
        if caches_state is None:
            _log(f"step={curr_step} | building caches(train)...", all_hosts=True)
            t0_caches = time.perf_counter()
            caches_state = data_config.build_caches("train", monitors=False)
            t1_caches = time.perf_counter()
            _log(f"built caches in {t1_caches - t0_caches:.3f}s; datasets={list(caches_state.keys())}")
        else:
            _log("reusing previously built caches")
        caches = caches_state

        # Determine datasets to evaluate
        if config.datasets is None:
            selected = list(caches.keys())
        else:
            selected = [name for name in config.datasets if name in caches]

        _log(f"step={curr_step} | selected datasets: {selected}")

        results = {}
        for ds_name in selected:
            ds_start_time = time.perf_counter()

            cache = caches[ds_name]
            input_store = cache.store.tree["input_ids"]  # type: ignore[index]

            # Persist selection once using fast offsets-based length checks
            if ds_name not in selected_indices_by_ds:
                need = int(max(1, config.num_documents))
                _log(f"dataset={ds_name} | selecting {need} doc(s) with length >= {config.chunk_size} via offsets")
                scan_t0 = time.perf_counter()
                num_rows = int(input_store.num_rows)
                selected_doc_indices: List[int] = []
                block = 4096
                idx = 0
                while idx < num_rows and len(selected_doc_indices) < need:
                    end = min(num_rows, idx + block)
                    offsets = input_store.offsets[idx : end + 1].read().result()
                    if idx == 0:
                        offsets = offsets.copy()
                        offsets[0] = 0
                    lens = offsets[1:] - offsets[:-1]
                    good = np.nonzero(lens >= int(config.chunk_size))[0]
                    for g in good:
                        sel_idx = idx + int(g)
                        selected_doc_indices.append(sel_idx)
                        if len(selected_doc_indices) >= need:
                            break
                    idx = end
                if len(selected_doc_indices) == 0:
                    _log(f"[WARN] dataset={ds_name} | no document with length >= {config.chunk_size} found; skipping")
                    continue
                selected_indices_by_ds[ds_name] = selected_doc_indices
                scan_t1 = time.perf_counter()
                _log(f"dataset={ds_name} | selected indices={selected_doc_indices} in {scan_t1 - scan_t0:.3f}s")
                # (Preview decode handled later with decode_once_state guard if configured)
            else:
                _log(f"dataset={ds_name} | reusing pinned indices={selected_indices_by_ds[ds_name]}")

            selected_doc_indices = selected_indices_by_ds[ds_name]
            # Read tokens for selected indices (minimally)
            selected_docs: List[np.ndarray] = []
            for _sel_idx in selected_doc_indices:
                try:
                    arr = np.asarray(input_store[_sel_idx], dtype=np.int32).reshape(-1)
                    selected_docs.append(arr)
                except Exception as e:
                    _log(f"[WARN] dataset={ds_name} | error reading doc {_sel_idx}: {e}")
                    continue

            # Use config.chunk_size, not model.Pos.size, to avoid evaluating mostly padding
            N = int(config.chunk_size)
            P = int(config.prompt_tokens if config.prompt_tokens is not None else N // 2)
            S = max(1, int(config.cursor_inc_tokens))

            if N != model.Pos.size:
                _log(f"[WARN] chunk_size={N} != model.Pos.size={model.Pos.size}, may cause recompilation")

            pz_values: List[float] = []
            span_ranges: List[tuple[int, int]] = []
            doc_indices_for_windows: List[int] = []
            first_mode_windows: List[np.ndarray] = []
            first_mode_indices: List[int] = []
            first_mode_doc_lens: List[int] = []
            first_eval_len: Optional[int] = None

            # Iterate over selected documents and compute windowed P(z)
            for doc_sel_idx, first_ids in zip(selected_doc_indices, selected_docs):
                # Determine document slice
                if config.doc_tokens is None:
                    eval_len = int(first_ids.shape[0])
                else:
                    eval_len = int(min(int(config.doc_tokens), int(first_ids.shape[0])))
                if first_eval_len is None:
                    first_eval_len = eval_len

                if eval_len == 0:
                    _log(f"[WARN] dataset={ds_name} | eval_len=0 for doc {doc_sel_idx}, skipping")
                    continue

                # Skip documents that are shorter than chunk_size (would be mostly padding)
                if eval_len < N:
                    _log(
                        f"[WARN] dataset={ds_name} | doc_idx={doc_sel_idx} doc_len={eval_len} < chunk_size={N}, skipping (would be mostly padding)"
                    )
                    continue

                doc_slice = first_ids[:eval_len]

                # Windowing mode selection
                _mode = (config.mode or "sliding").lower()
                if _mode == "first":
                    starts = [0]
                else:  # default to sliding
                    starts = list(range(0, max(eval_len - N, 0) + 1, S))
                    if not starts:
                        starts = [0]

                _log(
                    f"dataset={ds_name} | mode={_mode} doc_idx={doc_sel_idx} doc_len={eval_len} N={N} P={P} S={S} num_starts={len(starts)}"
                )

                # Time the window evaluations; first window usually includes compilation
                doc_t0 = time.perf_counter()
                first_window_time = None
                window_count = 0

                if _mode == "first":
                    # Collect the first window for batched compute later
                    s = 0
                    window = doc_slice[s : s + N]
                    if window.shape[0] < N:
                        pad_len = N - window.shape[0]
                        window = np.concatenate([window, np.full((pad_len,), pad_id, dtype=np.int32)], axis=0)
                    first_mode_windows.append(window)
                    first_mode_indices.append(doc_sel_idx)
                    first_mode_doc_lens.append(eval_len)
                    window_count = 1
                else:
                    for s in starts:
                        window = doc_slice[s : s + N]
                        if window.shape[0] < N:
                            pad_len = N - window.shape[0]
                            window = np.concatenate([window, np.full((pad_len,), pad_id, dtype=np.int32)], axis=0)
                        w_t0 = time.perf_counter()
                        lp = _compute_logprob_for_tokens(model, window, P)
                        p = float(np.exp(lp))
                        pz_values.append(p)
                        span_ranges.append((s, min(s + N - 1, eval_len - 1)))
                        doc_indices_for_windows.append(doc_sel_idx)
                        w_t1 = time.perf_counter()
                        window_count += 1
                        if first_window_time is None:
                            first_window_time = w_t1 - w_t0
                            _log(
                                f"dataset={ds_name} | doc_idx={doc_sel_idx} | first_window_time={first_window_time:.3f}s (includes possible compile)"
                            )
                        elif window_count % max(1, len(starts) // 5) == 0:
                            _log(
                                f"dataset={ds_name} | doc_idx={doc_sel_idx} | window {window_count}/{len(starts)} took {w_t1 - w_t0:.3f}s"
                            )

                doc_t1 = time.perf_counter()
                _log(
                    f"dataset={ds_name} | doc_idx={doc_sel_idx} | evaluated {window_count} windows in {doc_t1 - doc_t0:.3f}s"
                )

            # Aggregate and log scalars (only on process 0)
            # If first mode, perform batched evaluate across docs now
            if (config.mode or "sliding").lower() == "first" and len(first_mode_windows) > 0:
                arr2d = np.stack(first_mode_windows, axis=0)

                def _vmapped_batch(m: LmHeadModel, toks_2d: jnp.ndarray, prompt_len: int):
                    Pos = m.Pos.resize(N)

                    def single(tokens_1d: jnp.ndarray):
                        toks_named = hax.named(tokens_1d, Pos)
                        ex = LmExample.from_prompt_and_completion(
                            Pos, toks_named, prompt_length=int(prompt_len), ignore_id=pad_id
                        )
                        mm = m
                        if mp is not None:
                            mm = mp.cast_to_compute(mm)
                        logits = mm(ex.tokens, attn_mask=ex.attn_mask)
                        logits = logits.astype(jnp.float32)
                        nll = next_token_loss(
                            Pos=Pos,
                            Vocab=mm.Vocab,
                            logits=logits,
                            true_ids=ex.tokens,
                            loss_mask=ex.loss_mask,
                            reduction=None,
                        )
                        total_nll = hax.sum(nll, axis=Pos).array
                        return -total_nll

                    with hax.axis_mapping(axis_resources):
                        return jax.vmap(single, in_axes=0)(toks_2d)

                B = int(config.eval_batch_size) if config.eval_batch_size is not None else arr2d.shape[0]
                b0 = time.perf_counter()
                for i_b in range(0, arr2d.shape[0], B):
                    batch = jnp.asarray(arr2d[i_b : i_b + B], dtype=jnp.int32)
                    lp_vec = _vmapped_batch(model, batch, P)
                    for j, lp in enumerate(np.array(lp_vec)):
                        pz_values.append(float(np.exp(lp)))
                        idx = first_mode_indices[i_b + j]
                        doc_len = first_mode_doc_lens[i_b + j]
                        span_ranges.append((0, min(N - 1, doc_len - 1)))
                        doc_indices_for_windows.append(idx)
                b1 = time.perf_counter()
                _log(
                    f"dataset={ds_name} | batched first-mode forward for {arr2d.shape[0]} docs in {b1 - b0:.3f}s (batch_size={B})"
                )

            # Aggregate and log scalars (only on process 0)
            if len(pz_values) > 0:
                arr = np.asarray(pz_values, dtype=np.float64)
                mean_pz = float(np.mean(arr))
                median_pz = float(np.median(arr))
                max_pz = float(np.max(arr))
            else:
                mean_pz = median_pz = max_pz = 0.0

            ds_elapsed_time = time.perf_counter() - ds_start_time
            if jax.process_index() == 0:
                metrics = {
                    f"pz_eval/{ds_name}/num_windows": int(len(pz_values)),
                    f"pz_eval/{ds_name}/num_documents": (
                        int(len(set(doc_indices_for_windows))) if len(pz_values) > 0 else 0
                    ),
                    f"pz_eval/{ds_name}/mean_pz": mean_pz,
                    f"pz_eval/{ds_name}/median_pz": median_pz,
                    f"pz_eval/{ds_name}/max_pz": max_pz,
                    # Back-compat: log the first evaluated document length under doc_len
                    f"pz_eval/{ds_name}/doc_len": int(first_eval_len or 0),
                    f"pz_eval/{ds_name}/chunk_size": int(N),
                    f"pz_eval/{ds_name}/prompt_tokens": int(P),
                    f"pz_eval/{ds_name}/suffix_tokens": int(N - P),
                    f"pz_eval/{ds_name}/cursor_inc_tokens": int(S),
                    f"pz_eval/{ds_name}/eval_time_seconds": ds_elapsed_time,
                }
                _log(f"dataset={ds_name} | metrics: {metrics}")
                levanter.tracker.log(metrics, step=curr_step)
                _log(f"dataset={ds_name} | logged scalars (took {ds_elapsed_time:.2f}s)")

                # Log a value histogram of P(z) in [0, 1] with 10 bins of width 0.1
                # This mirrors the tracker Histogram used for entropy, but with fixed edges.
                if len(pz_values) > 0:
                    arr = jnp.asarray(pz_values, dtype=jnp.float32)
                    # Clip to [0, 1] for safety before binning
                    arr = jnp.clip(arr, 0.0, 1.0)
                    # Fixed 10 bins from 0.0 to 1.0 inclusive (edges length = 11)
                    edges = jnp.linspace(0.0, 1.0, 11, dtype=jnp.float32)
                    counts, edges_out = jnp.histogram(arr, bins=edges)
                    counts = counts.astype(jnp.int32)

                    # Populate Histogram fields
                    h_min = arr.min()
                    h_max = arr.max()
                    h_num = int(arr.size)
                    h_sum = arr.sum()
                    h_sum_sq = (arr**2).sum()
                    hist = Histogram(h_min, h_max, h_num, h_sum, h_sum_sq, edges_out, counts)

                    levanter.tracker.log({f"pz_eval/{ds_name}/pz_hist": hist}, step=curr_step)
                    _log(f"dataset={ds_name} | logged pz_hist with fixed [0,1] bins (10 bins)")

            # Artifacts
            if len(pz_values) > 0 and jax.process_index() == 0:
                if config.histogram and log_histogram_now:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        temp_hist_path = tmp_file.name
                    title = f"{ds_name} - P(z) over first doc"
                    if config.histogram_linear:
                        _ = create_pz_histogram_linear(
                            pz_list=np.asarray(pz_values),
                            threshold=config.pz_threshold,
                            save_path=temp_hist_path,
                            book_title=title,
                        )
                    else:
                        _ = create_pz_histogram(
                            pz_list=np.asarray(pz_values),
                            threshold=config.pz_threshold,
                            save_path=temp_hist_path,
                            book_title=title,
                        )
                    _log(f"dataset={ds_name} | logging histogram artifact")
                    levanter.tracker.current_tracker().log_artifact(
                        temp_hist_path, name=f"pz_hist_{ds_name}.png", type="plot"
                    )
                    pathlib.Path(temp_hist_path).unlink(missing_ok=True)

                if config.pz_npz:
                    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_npz:
                        np.savez(
                            tmp_npz.name,
                            pz_values=np.asarray(pz_values, dtype=np.float32),
                            span_ranges=np.asarray(span_ranges, dtype=np.int32),
                            doc_indices=np.asarray(doc_indices_for_windows, dtype=np.int32),
                            config_info=np.asarray([N, P, S], dtype=np.int32),
                        )
                        tmp_npz_path = tmp_npz.name
                    _log(f"dataset={ds_name} | logging NPZ artifact")
                    levanter.tracker.current_tracker().log_artifact(
                        tmp_npz_path, name=f"pz_values_{ds_name}.npz", type="data"
                    )
                    pathlib.Path(tmp_npz_path).unlink(missing_ok=True)

            # Preview decode once (use the first selected document)
            if config.decode_preview is not None and jax.process_index() == 0:
                state_key = f"{ds_name}__decoded"
                if not decode_once_state.get(state_key, False):
                    try:
                        preview_doc = selected_docs[0]
                        if config.doc_tokens is None:
                            preview_eval_len = int(preview_doc.shape[0])
                        else:
                            preview_eval_len = int(min(int(config.doc_tokens), int(preview_doc.shape[0])))
                        preview_len = int(min(int(config.decode_preview), preview_eval_len))
                        preview_ids = preview_doc[:preview_len].tolist()
                        preview_text = tokenizer.decode(preview_ids, skip_special_tokens=False)
                        meta = {
                            f"pz_eval/{ds_name}/preview_text": preview_text,
                            f"pz_eval/{ds_name}/preview_token_sum": int(sum(preview_ids)),
                        }
                        if config.verify_treecache:
                            meta[f"pz_eval/{ds_name}/first_doc_len"] = int(preview_doc.shape[0])
                        levanter.tracker.log(meta, step=curr_step)
                        decode_once_state[state_key] = True
                    except Exception:
                        pass

            results[ds_name] = True

        eval_total_time = time.time() - eval_start_time
        if jax.process_index() == 0:
            levanter.tracker.log(
                {
                    "pz_eval/total_eval_time_seconds": eval_total_time,
                    "pz_eval/num_datasets_evaluated": len(results),
                },
                step=curr_step,
            )
            _log(f"total P(z) eval time: {eval_total_time:.2f}s for {len(results)} dataset(s)")

        return results

    def cb(step: StepInfo, force=False):
        if step.step == 0 and not force:
            return
        # CRITICAL FIX: Do NOT early-return on non-zero processes!
        # The model evaluation involves JAX collectives that require ALL hosts to participate.
        # Returning early on non-zero processes causes a deadlock.
        _log(f"entering P(z) callback for step={int(step.step)}", all_hosts=True)
        model = step.eval_model

        # Only process 0 logs to tracker
        if jax.process_index() == 0:
            # Heartbeat metric to confirm W&B visibility at each callback tick
            levanter.tracker.log({"pz_eval/heartbeat": 1}, step=int(step.step))
            _log("logged heartbeat=1")

        log_hist = True
        if config.histogram and config.histogram_every_steps is not None:
            try:
                log_hist = step.step % int(config.histogram_every_steps) == 0
            except Exception:
                log_hist = False

        # Wrap in try-except to surface errors instead of silent hangs
        try:
            t0 = time.perf_counter()
            _run_for_model(model, log_histogram_now=log_hist, curr_step=int(step.step))
            t1 = time.perf_counter()
            _log(f"finished P(z) callback in {t1 - t0:.3f}s")
        except Exception as e:
            # Print error with process index for debugging multi-host issues
            _log(f"ERROR: {e}")
            _log(f"Traceback:\n{traceback.format_exc()}")
            raise

    return cb
