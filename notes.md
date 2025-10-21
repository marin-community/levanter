# Inference Notes

Let's start from our fixed points. We're doing auto-regressive decoding over a
Llama (for now) model, and we are limited by the interface to the
ragged-paged-attention kernel. That kernel looks like this:

```python
@named_call
def ragged_paged_attention(
    q: NamedArray,  # [Tok, KVHeads, QHeadsPerGroup, HeadSize]
    kv_pages: NamedArray,  # [Page, PageSize, 2 * KVHeads, HeadDim]
    kv_lens: NamedArray,  # i32[Seq]
    page_indices: NamedArray,  # i32[Seq, PagePerSeq]
    cu_q_lens: NamedArray,  # i32[Seq + 1] <-- cumulative lengths for the sequences, including new tokens
    num_seqs: jnp.ndarray,
    sm_scale: float = 1.0,
    soft_cap: float | None = None,
) -> NamedArray:

        attn_tokens = ragged_paged_attention(
            q,
            kv_cache.kv_pages,
            batch_info.seq_lens,
            batch_info.page_indices,
            batch_info.cu_q_lens,
            batch_info.num_seqs,
            sm_scale=sm_scale,
            soft_cap=self.config.logits_soft_cap,
        )
```

Key attributes here are that:

* Ragged paged attention operates on _pages_ at a time, so we need to work with that
* Pages are separately indexed by page_indices, presumably mapping [seq, offset_of_page]
* Query does _not_ have a sequence dimension. Instead we're flattened for some reason, presumably to allow different sequence lengths.


_We assume pages are fully packed!_
* The page_indices & kv_lens must be symmetric with where we _write_ our page updates in new_token_dests

There's a few implications for this:

We can't be "too dumb" in our allocation strategy for pages and sequences.  We need to ensure our page descriptions are always fully packed.
Since sequences are independent, we _can_ and should assign a different page for each sequence and slot.
We should be able to statically assign the maximal set of pages ahead of time, since we index on the kv_lens and not the pages, those are just a static lookup


- We don't care about this for RL, but important for serving I suppose.

The code flow from here is:

-> Attention::paged_decode
-> LLama::decode
```
    @named_call
    def decode(
        self,
        x: NamedArray,
        kv_cache: KvPageCache,
        batch_info: DecodeState,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, KvPageCache]:
    ```
-> Engine::_run_generation_loop
        ```logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)```

So key things here:

* at Llama call time we now have a "KVPageCache" which we call update on to write out our new keys

```
        kv_cache = kv_cache.update(batch_info, k, v)
```

That _assumes_ everything is already writable and writes into the approrpriate page in the weird JAX interleaved format:

```

        K = jnp.asarray(batch_info.num_new_tokens, jnp.int32)
        t_pages, t_slots = batch_info.pages_and_slots()  # [T] int32 (first K valid)

        updated = kv_update_unified_prefix(
            self.kv_pages.array,
            t_pages.astype(jnp.int32).array,
            t_slots.astype(jnp.int32).array,
            new_k.array,
            new_v.array,
            K,
        )
```


## Questions

* Can we lift our KV cache management up to the CPU/top-level?
* What would we need to change to have per-sequence attention masks and e.g. left-pad pre-fill so we can allocate a page at a time?
* Can we use the JAX KV format directly?
