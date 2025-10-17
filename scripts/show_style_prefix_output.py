import json
from pathlib import Path

from levanter.data.text import ChatLmDatasetFormat, StylePrefixConfig, preprocessor_for_format
from levanter.compat.hf_checkpoints import load_tokenizer


def main():
    dataset_path = Path("data/style_demo/train.jsonl")
    entries = [json.loads(line) for line in dataset_path.open()]
    tokenizer = load_tokenizer("gpt2")

    chat_template = (
        "{%- for message in messages -%}\n"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt %}assistant:{% endif %}"
    )

    format_cfg = ChatLmDatasetFormat(
        messages_field="messages",
        single_turn=False,
        chat_template=chat_template,
        pack=True,
        mask_user_turns=False,
        style_prefix=StylePrefixConfig(
            prefix_token="<style>",
            suffix_token="</style>",
            style_field="style",
        ),
    )

    processor = preprocessor_for_format(format_cfg, tokenizer)
    processed = processor(entries[:2])

    for idx, example in enumerate(processed):
        print(f"Example {idx}")
        print("input_ids:", example["input_ids"][:20])
        print("assistant_masks:", example["assistant_masks"][:20])
        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"][:20])
        print("tokens:", tokens)
        print()


if __name__ == "__main__":
    main()
