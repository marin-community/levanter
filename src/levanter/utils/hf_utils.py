import os

from levanter.logging import silence_transformer_nag
from levanter.utils.py_utils import logical_cpu_core_count


silence_transformer_nag()
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # noqa: E402


HFTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def num_cpus_used_by_tokenizer(tokenizer) -> int:
    if getattr(tokenizer, "is_fast", False):
        if os.getenv("TOKENIZERS_PARALLELISM", "true").lower() in _HF_TOKENIZER_OFF_VALUES:
            return 1
        else:
            # This is a bit hacky, but HF's fast tokenizers are parallelized under the hood.
            # we reserve a couple of cores just so Ray has somewhere to run the coordinator.
            return min(max(1, logical_cpu_core_count() - 2), 32)
    else:
        return 1


_HF_TOKENIZER_OFF_VALUES = {"off", "false", "f", "no", "n", "0"}
