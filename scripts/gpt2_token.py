from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Suppose you have token IDs
token_ids = [3198, 1110, 11, 257, 1310, 2576, 3706, 20037, 1043, 257, 17598, 287]
# Decode them to string
decoded_text = tokenizer.decode(token_ids)
print("Decoded:", decoded_text)

token_ids = [50256, 1881, 1110, 11, 257, 1310, 2576, 3706, 20037, 1043, 257, 17598, 287]
# Decode them to string
decoded_text = tokenizer.decode(token_ids)
print("Decoded:", decoded_text)


# Optionally, tokenize a string to get token IDs
text = '''One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt.

Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt."
'''
encoded_ids = tokenizer.encode(text)
print('Token length: ', len(encoded_ids))
print("Encoded:", encoded_ids)