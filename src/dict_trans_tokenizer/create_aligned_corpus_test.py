from transformers import AutoTokenizer, PreTrainedTokenizerFast


class TestClass:
    def load_tokenizer(self):
        tokenizer_name = "roberta-base"
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        return tokenizer

    def test_merge_tokens(self):
        from dict_trans_tokenizer.create_aligned_corpus import get_prefix, merge_tokens

        tokenizer = self.load_tokenizer()
        prefixes = get_prefix(tokenizer)
        text = "I read a book about HistoricalLinguisitics"
        tokenized_text = " ".join(
            tokenizer.convert_ids_to_tokens(
                tokenizer.encode(text, add_special_tokens=False)
            )
        )
        merged = merge_tokens(*prefixes, tokenized_text)
        assert len(merged.split()) == 6

    def test_tokenize_dictionary(self):
        from dict_trans_tokenizer.create_aligned_corpus import (
            AlignmentMode,
            tokenize_dictionary,
        )

        source_tokenizer = self.load_tokenizer()
        target_tokenizer = self.load_tokenizer()
        target_word = "HistoricalLinguistics"
        definition = ["the study of language change over time"]
        tokenized_corpus = tokenize_dictionary(
            target_word,
            definition,
            source_tokenizer,
            target_tokenizer,
            AlignmentMode.WORD,
        )
        assert len(tokenized_corpus) == 1
        assert len(tokenized_corpus[0].split()) == 9
