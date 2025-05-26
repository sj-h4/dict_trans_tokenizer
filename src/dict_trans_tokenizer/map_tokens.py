import math
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
)

from dict_trans_tokenizer.types.alignment_mode import AlignmentMode
from dict_trans_tokenizer.utils.logging_config import get_logger

logger = get_logger(__name__)


class TokenMpper:
    """
    Derived from <https://github.com/LAGoM-NLP/transtokenizer/blob/99a41391da7b6045e80cc6f49bf7cf23abca9056/transtokenizers/transtokenizers.py#L273>

    Attributes:
        tokenized_possible_translations: [TODO:attribute]
        untokenized_possible_translations: [TODO:attribute]
        OLD_TOKENIZER_1ST_PREFIX: [TODO:attribute]
        NEW_TOKENIZER_1ST_PREFIX: [TODO:attribute]
        OLD_TOKENIZER_2ND_PREFIX: [TODO:attribute]
        NEW_TOKENIZER_2ND_PREFIX: [TODO:attribute]
    """

    def __init__(self, space_token: str = "Ġ"):
        """

        Args:
            space_token: Token for space. Default is "Ġ"
        """
        self.tokenized_possible_translations: defaultdict[
            str, defaultdict[str, int]
        ] = defaultdict(lambda: defaultdict(int))
        self.untokenized_possible_translations: defaultdict[
            str, defaultdict[str, int]
        ] = defaultdict(lambda: defaultdict(int))
        self.OLD_TOKENIZER_1ST_PREFIX = space_token
        self.NEW_TOKENIZER_1ST_PREFIX = space_token

        # HACK: This is a workaround for the case where BPE is used as a tokenizer.
        self.OLD_TOKENIZER_2ND_PREFIX = ""
        self.NEW_TOKENIZER_2ND_PREFIX = ""

    def add_token_pair(self, count, new_token, old_token):
        self.tokenized_possible_translations[new_token][old_token] += count

    def add_word_pair(self, count, new_word, old_word, all_to_all_mapping=False):
        # tokenize the words
        # (recall that we use the long hyphen to replace spaces inside words, to merge the tokens again)
        old_word_tokenized = old_word.split("—")
        new_word_tokenized = new_word.split("—")

        # if the token list dont have the same length, compute the smallest common multiple of their lengths
        if all_to_all_mapping:
            count_dilution = len(old_word_tokenized)
            old_word_tokenized = np.tile(old_word_tokenized, len(new_word_tokenized))
            new_word_tokenized = np.repeat(new_word_tokenized, count_dilution)
        elif len(old_word_tokenized) != len(new_word_tokenized):
            gcd = math.gcd(len(old_word_tokenized), len(new_word_tokenized))
            count_dilution = len(old_word_tokenized) // gcd
            old_word_tokenized = np.repeat(
                old_word_tokenized, len(new_word_tokenized) // gcd
            )
            new_word_tokenized = np.repeat(new_word_tokenized, count_dilution)
        else:
            gcd = 1
            count_dilution = 1

        # perform this operation for each token pair in the word
        for token_old, token_new in zip(
            old_word_tokenized, new_word_tokenized, strict=False
        ):
            # add the translation to the dictionary
            self.tokenized_possible_translations[token_new][token_old] += max(
                1, count // count_dilution
            )

    def map_special_tokens(
        self,
        new_tokenizer: PreTrainedTokenizerFast,
        old_tokenizer: PreTrainedTokenizerFast,
    ):
        new_tokenizer_vocab = set(new_tokenizer.vocab.keys())
        old_tokenizer_vocab = set(old_tokenizer.vocab.keys())
        # add a mapping for all numbers
        for i in range(9999):
            str_i = str(i)
            if str_i in new_tokenizer_vocab:
                self.add_token_pair(
                    1,
                    str_i,
                    str_i
                    if str_i in old_tokenizer_vocab
                    else old_tokenizer.tokenize(str_i)[0],
                )
            if len(new_tokenizer.tokenize(str_i)) == 1:
                self.add_token_pair(
                    1,
                    new_tokenizer.tokenize(str_i)[0],
                    old_tokenizer.tokenize(str_i)[0],
                )
            if len(new_tokenizer.tokenize(" " + str_i)) == 1:
                self.add_token_pair(
                    1,
                    new_tokenizer.tokenize(" " + str_i)[0],
                    old_tokenizer.tokenize(" " + str_i)[0],
                )
        for i in range(99):
            str_i = "0" + str(i)
            if str_i in new_tokenizer_vocab:
                self.add_token_pair(
                    1,
                    str_i,
                    str_i
                    if str_i in old_tokenizer_vocab
                    else old_tokenizer.tokenize(str_i)[0],
                )
            if len(new_tokenizer.tokenize(str_i)) == 1:
                self.add_token_pair(
                    1,
                    new_tokenizer.tokenize(str_i)[0],
                    old_tokenizer.tokenize(str_i)[0],
                )
            if len(new_tokenizer.tokenize(" " + str_i)) == 1:
                self.add_token_pair(
                    1,
                    new_tokenizer.tokenize(" " + str_i)[0],
                    old_tokenizer.tokenize(" " + str_i)[0],
                )

        # add a mapping for all punctuation (and words that exist in both models)
        for token in new_tokenizer_vocab:
            ## skip if any token char is a letter or digit
            # if any(c.isalnum() for c in token): continue
            token2 = token
            # replace the start symbol of the new model with the one of the old model
            if (
                self.NEW_TOKENIZER_1ST_PREFIX != ""
                or self.OLD_TOKENIZER_1ST_PREFIX != ""
            ):
                token2 = token.replace(
                    self.NEW_TOKENIZER_1ST_PREFIX, self.OLD_TOKENIZER_1ST_PREFIX
                )
            # replace the continuation symbol of the new model with the one of the old model
            if (
                self.NEW_TOKENIZER_2ND_PREFIX != ""
                or self.OLD_TOKENIZER_2ND_PREFIX != ""
            ):
                token2 = token2.replace(
                    self.NEW_TOKENIZER_2ND_PREFIX, self.OLD_TOKENIZER_2ND_PREFIX
                )
            # skip if token is not in the old model
            if token2 not in old_tokenizer_vocab:
                continue
            # add the mapping
            self.tokenized_possible_translations[token][token2] += 1

        def or_old_unk_token(token, fallback_token=None):
            if (token is not None) and (token in old_tokenizer_vocab):
                return token
            if (fallback_token is not None) and (fallback_token in old_tokenizer_vocab):
                return fallback_token
            return old_tokenizer.unk_token

        # add a mapping for special tokens (i.e. pad, unk, bos, eos, sep, cls, mask)
        very_large_number = 1_000_000_000
        if new_tokenizer.pad_token is not None:
            self.add_token_pair(
                very_large_number,
                new_tokenizer.pad_token,
                or_old_unk_token(old_tokenizer.pad_token),
            )
        if new_tokenizer.unk_token is not None:
            self.add_token_pair(
                very_large_number,
                new_tokenizer.unk_token,
                or_old_unk_token(old_tokenizer.unk_token),
            )
        if new_tokenizer.bos_token is not None:
            self.add_token_pair(
                very_large_number,
                new_tokenizer.bos_token,
                or_old_unk_token(old_tokenizer.bos_token, old_tokenizer.cls_token),
            )
        if new_tokenizer.eos_token is not None:
            self.add_token_pair(
                very_large_number,
                new_tokenizer.eos_token,
                or_old_unk_token(old_tokenizer.eos_token, old_tokenizer.sep_token),
            )
        if new_tokenizer.cls_token is not None:
            self.add_token_pair(
                very_large_number,
                new_tokenizer.cls_token,
                or_old_unk_token(old_tokenizer.cls_token, old_tokenizer.bos_token),
            )
        if new_tokenizer.sep_token is not None:
            self.add_token_pair(
                very_large_number,
                new_tokenizer.sep_token,
                or_old_unk_token(old_tokenizer.sep_token, old_tokenizer.eos_token),
            )
        if new_tokenizer.mask_token is not None:
            self.add_token_pair(
                very_large_number,
                new_tokenizer.mask_token,
                or_old_unk_token(old_tokenizer.mask_token, old_tokenizer.pad_token),
            )

    def map_tokens(
        self,
        mapped_tokens_file: str,
        source_tokenizer: PreTrainedTokenizerFast,
        target_tokenizer: PreTrainedTokenizerFast,
        min_count_request_for_consideration: int,
        alignment_mode: AlignmentMode,
    ) -> tuple[list[str], list[str]]:
        """Map tokens from source to target language.

        Args:
            mapped_tokens_file: Path to the mapped tokens file
            source_tokenizer: Source language tokenizer
            target_tokenizer: Target language tokenizer
            min_count_request_for_consideration: Minimum count for consideration
            alignment_mode: Alignment mode (word or token)

        Returns:
            Tuple of (mapped tokens, mapped source tokens)
        """
        logger.info(
            f"Starting token mapping with min count: {min_count_request_for_consideration}"
        )
        logger.info(f"Alignment mode: {alignment_mode}")

        old_tokenizer = source_tokenizer
        new_tokenizer = target_tokenizer

        # save the vocabularies in a set for improved performance
        old_tokenizer_vocab = set(old_tokenizer.vocab.keys())
        new_tokenizer_vocab = set(new_tokenizer.vocab.keys())

        mapped_target_tokens = set()
        mapped_source_tokens = set()

        total_alignments = 0
        with open(mapped_tokens_file) as f:
            for _line in f:
                total_alignments += 1

        with open(mapped_tokens_file) as f:
            for line in tqdm(f, total=total_alignments):
                # remove the newline character
                line = line.rstrip("\n")
                # skip empty lines
                if line == "":
                    continue
                # split the line on the tab character
                old_word, new_word, _, count = line.split("\t")
                # reject <eps> mappings
                if old_word == "<eps>":
                    continue
                if new_word == "<eps>":
                    continue
                # convert the count to an integer
                count = math.ceil(float(count))

                if new_word == "Ġhrung":
                    print("*********")
                    print(new_word, count)
                    print("*********")

                # skip pairs that happened rarely (likely noise)
                # if count < min_count_request_for_consideration:
                if count < min_count_request_for_consideration:
                    continue
                # add the token pair to the token dictionary
                if (alignment_mode != AlignmentMode.WORD) or (
                    (new_word in new_tokenizer_vocab)
                    and (old_word in old_tokenizer_vocab)
                ):
                    self.add_token_pair(count, new_word, old_word)
                else:
                    half_count = max(1, count // 2)
                    self.add_word_pair(
                        half_count, new_word, old_word, all_to_all_mapping=True
                    )
                    self.add_word_pair(
                        half_count, new_word, old_word, all_to_all_mapping=False
                    )
                mapped_target_tokens.add(new_word)
                mapped_source_tokens.add(old_word)
                # add the word translation to the dictionary (for diagnostics purposes only)
                self.untokenized_possible_translations[new_word][old_word] += count

        # check how many tokens have a translation, compared to the total number of tokens
        print(f"Number of tokens with a translation: {len(mapped_target_tokens)}")
        print(f"Number of vocab: {len(new_tokenizer)}")
        print(
            f"Percentage of tokens with a translation: {int(len(mapped_target_tokens) / len(new_tokenizer) * 1000) / 10}%"
        )
        print(
            f"Total numner of tokens with a translation: {len(self.tokenized_possible_translations)}"
        )

        return list(mapped_target_tokens), list(mapped_source_tokens)
