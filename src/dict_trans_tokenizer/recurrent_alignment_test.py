from transformers import AutoTokenizer


def test_delete_tokens_from_tokenizer():
    from dict_trans_tokenizer.recurrent_alignment import delete_tokens_from_tokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Original tokenizer vocab size: ", len(tokenizer))
    original_tokenized = tokenizer.tokenize("percent")
    print(original_tokenized)
    assert original_tokenized == ["percent"]
    assert len(tokenizer) == 50257

    unwanted_tokens = ["percent"]
    deleted_tokenizer = delete_tokens_from_tokenizer(tokenizer, unwanted_tokens)
    print("Deleted tokenizer vocab size: ", len(deleted_tokenizer))
    deleted_tokenized = deleted_tokenizer.tokenize("percent")
    print(deleted_tokenized)
    assert deleted_tokenized == ["per", "cent"]
    assert len(deleted_tokenizer) == 50256

    unwanted_tokens = ["per", "cent"]
    deleted_tokenizer = delete_tokens_from_tokenizer(tokenizer, unwanted_tokens)
    deleted_tokenized = deleted_tokenizer.tokenize("percent")
    print(deleted_tokenized)

    unwanted_tokens = ["p", "erc", "ent"]
    deleted_tokenizer = delete_tokens_from_tokenizer(tokenizer, unwanted_tokens)
    deleted_tokenized = deleted_tokenizer.tokenize("percent")
    print(deleted_tokenized)
