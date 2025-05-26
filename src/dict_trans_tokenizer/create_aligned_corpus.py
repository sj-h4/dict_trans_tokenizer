import os
import re

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from dict_trans_tokenizer.types.alignment_mode import AlignmentMode
from dict_trans_tokenizer.types.bilingual_dict import BilingualDict
from dict_trans_tokenizer.utils.logging_config import get_logger

logger = get_logger(__name__)


def merge_tokens(
    first_token_prefix: str, second_token_prefix: str, tokenized_text: str
) -> str:
    """Merge tokens from word units.
    This helps to obtain a better alignment. See <http://arxiv.org/abs/2408.04303>

    Args:
        first_token_prefix: [TODO:description]
        second_token_prefix: [TODO:description]
        tokenized_text: [TODO:description]

    Returns:
        merged tokens sequence
    """
    pattern = rf"(?!{first_token_prefix})([^\W\d_])[ ](?!{first_token_prefix})(?={second_token_prefix}[^\W\d_])"
    return re.sub(pattern, r"\1—", tokenized_text)


def get_prefix(
    tokenizer: PreTrainedTokenizerFast, first_token_prefix="Ġ", second_token_prefix=""
) -> tuple[str, str]:
    """Get the prefix of the first and second token in the tokenizer.

    Args:
        tokenizer: [TODO:description]

    Returns:
        first_token_prefix, second_token_prefix
    """
    token = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(" a", add_special_tokens=False)[0]
    )
    if isinstance(token, list):
        token = token[0]
    first_token_prefix = token.rstrip("a")
    # HACK: This is a workaround for specific cases.
    second_token_prefix = ""
    # second_token_prefix = tokenizer.convert_ids_to_tokens(
    #     tokenizer.encode("aaaaaaaaaaaaaaaaaaaaaaaaaaa", add_special_tokens=False)[1]
    # ).rstrip("a")
    return first_token_prefix, second_token_prefix


def tokenize_dictionary(
    target_word: str,
    definition: list[str],
    source_tokenizer: PreTrainedTokenizerFast,
    target_tokenizer: PreTrainedTokenizerFast,
    alignment_mode: AlignmentMode,
    first_token_prefix="Ġ",
    second_token_prefix="",
) -> list[str]:
    """Tokenize the entry and definition of the dictionary.

    Args:
        target_word: [TODO:description]
        definition: [TODO:description]
        source_tokenizer: [TODO:description]
        target_tokenizer: [TODO:description]
        alignment_mode: [TODO:description]
        first_token_prefix: The prefix of the first token (default: "Ġ")
        second_token_prefix: The prefix of the second token (default: "")

    Returns:
        tokenized pair of source (definition) and target (word entry)
    """
    tokenized_corpus = []
    tokenized_word = " ".join(
        target_tokenizer.convert_ids_to_tokens(
            target_tokenizer.encode(target_word.strip(), add_special_tokens=False)
        )
    )
    if len(tokenized_word) == 0:
        return tokenized_corpus

    tokenized_definition = [
        " ".join(
            source_tokenizer.convert_ids_to_tokens(
                source_tokenizer.encode(d.strip(), add_special_tokens=False)
            )
        )
        for d in definition
    ]
    if any(len(d) == 0 for d in tokenized_definition):
        return tokenized_corpus

    match alignment_mode:
        case AlignmentMode.WORD:
            target = merge_tokens(
                first_token_prefix, second_token_prefix, tokenized_word
            )
            sources = [
                merge_tokens(first_token_prefix, second_token_prefix, d)
                for d in tokenized_definition
            ]
        case AlignmentMode.TOKEN:
            target = tokenized_word
            sources = tokenized_definition
    for source in sources:
        tokenized_corpus.append(source.strip() + " ||| " + target.strip())

    return tokenized_corpus


def create_aligned_corpus(
    source_tokenizer: PreTrainedTokenizerFast,
    target_tokenizer: PreTrainedTokenizerFast,
    dictionary: list[BilingualDict],
    output_path: str,
    alignment_mode: AlignmentMode,
) -> str:
    """Create an aligned corpus for token alignment.

    Args:
        source_tokenizer: Source language tokenizer
        target_tokenizer: Target language tokenizer
        dictionary: List of bilingual dictionary entries
        output_path: Path to save the aligned corpus
        alignment_mode: Alignment mode (word or token)

    Returns:
        Path to the created aligned corpus
    """
    logger.info("Creating aligned corpus")
    logger.info(f"Dictionary size: {len(dictionary)}")
    logger.info(f"Alignment mode: {alignment_mode}")

    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        logger.warning(f"Output file already exists: {output_path}")
        return output_path

    tokenized_corpus: list[str] = []
    for entry in tqdm(dictionary):
        target_word = entry.entry.lower().strip()
        definition = entry.definitions
        if len(target_word) == 0:
            print("Empty target word")
            continue
        if len(definition) == 0:
            print(f"Empty definition for {target_word}")
            continue

        tokenized_dict = tokenize_dictionary(
            target_word,
            definition,
            source_tokenizer,
            target_tokenizer,
            alignment_mode,
        )
        if len(tokenized_dict) == 0:
            continue
        tokenized_corpus.extend(tokenized_dict)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(tokenized_corpus))
    return output_path
