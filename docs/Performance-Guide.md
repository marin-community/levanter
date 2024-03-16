# Introduction

This is the very beginnings of a performance guide for Levanter. It's currently mostly a collection of notes and ideas,
but it will eventually be a comprehensive guide to optimizing Levanter (and potentially other JAX programs).



See also the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)

# Profiling

## Enabling the Profiler

Levanter uses JAX's built-in profiler. You can enable it by adding the `--trainer.profiler true` flag
to the command line. This will generate a trace file in the `./logs` directory, under `./logs/<run_id>/profiler/plugins/profile/<datetime>`.
(Yeah, it's a mess, but it's what JAX wants to do.)
It will also upload the information to the relevant tracker (such as Weights & Biases or TensorBoard).

Here are the full list of profiling related options:

| Argument                           | Description | Default |
|------------------------------------|-------------|---------|
| `--trainer.profiler`               | Enable the profiler | `false` |
| `--trainer.profiler_start_step`    | The step to start profiling | `5`     |
| `--trainer.profiler_num_steps`     | The number of steps to profile | `100`   |
| `--trainer.profiler_perfetto_link` | Whether to generate a Perfetto URL | `false` |

As usual, these can be specified in the yaml configuration file as well.

In a multi-process setup, each node will save a profile, but only the first node will upload it to the tracker.
All of them will be available in the `./logs` directory (on each node).


## Examining a Profile

See the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html) for more information on how to examine a profile.

JAX offers two main ways to examine a profile: TensorBoard and Perfetto. I find the Perfetto trace viewer
to be basically useless. Perhaps I just don't understand it.

### TensorBoard

TensorBoard is pretty good. You want to download the trace files (e.g. `plugins/profile/2024_03_16_07_26_24`)
and run `tensorboard --logdir <dir>` where `<dir>` is the *directory containing plugins* (not the plugins directory itself).
Then you can navigate to http://localhost:6006/#profile in your browser and see the profile.

There are three sections I find particularly useful:

1. The overview page tells you MMU utilization and the top 10 operations.
2. **op_profile** shows you the time spent in each operation (by type). You end up with annoying names like `fusion.1772`,
but with some patience and work you can back those out by looking at the next section (under XLA Ops).
3. **trace_viewer** shows you the actual trace of operations as a big timeline. It takes a long time to load.

### Perfetto

But as a rough cut, you can use [Perfetto](https://ui.perfetto.dev/) in Chrome (Firefox doesn't work super well with it)
to examine the `perfetto_trace.json.gz` file.


## Interpreting JAX terms in profiles

* `jvp(OP)` means the forward pass. (JVP stands for Jacobian-vector product.)
* `transpose(jvp(OP))` means the backward pass.
* `remat` (short for rematerialization) means that the operation is recomputed in the backward pass, i.e. gradient checkpointing.
