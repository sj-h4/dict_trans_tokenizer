import argparse
import copy
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from transtokenizers import align

from dict_trans_tokenizer.create_aligned_corpus import create_aligned_corpus
from dict_trans_tokenizer.map_tokens import TokenMpper
from dict_trans_tokenizer.remap_model import remap_model
from dict_trans_tokenizer.types.alignment_mode import AlignmentMode
from dict_trans_tokenizer.types.bilingual_dict import BilingualDict
from dict_trans_tokenizer.utils.load_bilingual_dict import load_bilingual_dict
from dict_trans_tokenizer.utils.logging_config import get_logger, setup_logging
from dict_trans_tokenizer.utils.set_seed import set_seed

logger = get_logger(__name__)


def delete_tokens_from_tokenizer(
    tokenizer: PreTrainedTokenizerFast, unwanted_tokens: list[str]
) -> PreTrainedTokenizerFast:
    """Delete unwanted tokens from the tokenizer.

    Derived from <https://github.com/1kkiRen/Tokenizer-Changer.git>

    Args:
        tokenizer: Tokenizer to delete tokens from
        unwanted_tokens: tokens to be removed

    Returns:
        Tokenizer with unwanted tokens removed
    """
    original_tokenizer = copy.deepcopy(tokenizer)
    tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())

    # Create a set of unwanted tokens for faster lookup
    unwanted_set = set(unwanted_tokens)

    logger.debug(f"unwanted tokens: {len(unwanted_set)}")
    logger.debug(f"old tokenizer vocab size: {len(tokenizer_json['model']['vocab'])}")

    # Remove tokens from vocabulary
    tokenizer_json["model"]["vocab"] = {
        token: idx
        for token, idx in tokenizer_json["model"]["vocab"].items()
        if token not in unwanted_set
    }

    # ReIndex the vocab
    tokenizer_json["model"]["vocab"] = {
        token: idx
        for idx, (token, _) in enumerate(tokenizer_json["model"]["vocab"].items())
    }

    logger.debug(f"new tokenizer vocab size: {len(tokenizer_json['model']['vocab'])}")

    # If the tokenizer uses merges (BPE), filter them as well
    if "merges" in tokenizer_json["model"]:
        filtered_merges = []
        for merge_tokens in tokenizer_json["model"]["merges"]:
            merged_token = "".join(merge_tokens)
            if (merged_token not in unwanted_set) and (
                not any(token in unwanted_set for token in merge_tokens)
            ):
                filtered_merges.append(merge_tokens)
        tokenizer_json["model"]["merges"] = filtered_merges

    new_backend_tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    new_tokenizer = original_tokenizer.__class__(
        tokenizer_object=new_backend_tokenizer, **original_tokenizer.init_kwargs
    )
    return new_tokenizer


def align_tokens(
    source_tokenizer: PreTrainedTokenizerFast,
    target_tokenizer: PreTrainedTokenizerFast,
    dictionary: list[BilingualDict],
    corpus_path: str,
    fast_align_path: str,
    token_mapper: TokenMpper,
    alignment_mode: AlignmentMode,
    alignment_log: list[list[str | int]],
    min_count_request_for_consideration: int = 0,
    iteration: int = 0,
    loop: int = 0,
) -> tuple[PreTrainedTokenizerFast, TokenMpper, list[list[str | int]]]:
    """Align tokens in the dictionary.
    Args:
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        dictionary: Dictionary to align
        corpus_path: Path to the corpus to align
        fast_align_path: Path to the fast align binary
        token_mapper: Token mapper
        alignment_mode: Alignment mode
        iteration: The number of iterations to run the alignment process
    Returns:
    """
    aligned_corpus = create_aligned_corpus(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        dictionary=dictionary,
        output_path=corpus_path,
        alignment_mode=alignment_mode,
    )

    mapped_tokens_file = align(
        aligned_corpus,
        fast_align_path=fast_align_path,
    )

    new_mapper = TokenMpper()
    new_mapper.tokenized_possible_translations.update(
        token_mapper.tokenized_possible_translations
    )

    new_mapped_tokens, new_mapped_source_tokens = new_mapper.map_tokens(
        mapped_tokens_file,
        source_tokenizer,
        target_tokenizer,
        min_count_request_for_consideration,
        alignment_mode,
    )

    # Log the alignment progress
    print(f"Loop {loop}: {len(new_mapped_tokens)} tokens were mapped.")
    print(f"Min count: {min_count_request_for_consideration}")
    logger.info(f"Loop {loop}: {len(new_mapped_tokens)} tokens were mapped.")
    logger.info(f"Min count: {min_count_request_for_consideration}")
    logger.info(f"tokenizer vocab size: {len(target_tokenizer)}")

    alignment_log.append(
        [
            iteration,
            loop,
            len(new_mapped_tokens),
            len(new_mapper.tokenized_possible_translations),
            min_count_request_for_consideration,
        ]
    )

    if len(new_mapped_tokens) == 0:
        print(f"Alignment completed in {loop} loops.")
        print(f"{len(new_mapped_tokens)} tokens were mapped.")
        logger.info(f"{len(new_mapped_tokens)} tokens were mapped.")
        return target_tokenizer, new_mapper, alignment_log
    else:
        loop += 1
        deleted_tokenizer = delete_tokens_from_tokenizer(
            target_tokenizer, new_mapped_tokens
        )

        next_corpus_path = (
            corpus_path.split(".")[0] + f".iteration_{iteration}.loop_{loop}.moses"
        )
        return align_tokens(
            source_tokenizer=source_tokenizer,
            target_tokenizer=deleted_tokenizer,
            dictionary=dictionary,
            corpus_path=next_corpus_path,
            fast_align_path=fast_align_path,
            token_mapper=new_mapper,
            alignment_mode=alignment_mode,
            alignment_log=alignment_log,
            min_count_request_for_consideration=min_count_request_for_consideration,
            loop=loop,
            iteration=iteration,
        )


def iterate_alignment(
    source_tokenizer: PreTrainedTokenizerFast,
    target_tokenizer: PreTrainedTokenizerFast,
    dictionary: list[BilingualDict],
    corpus_path: str,
    fast_align_path: str,
    token_mapper: TokenMpper,
    alignment_mode: AlignmentMode,
    alignment_log: list[list[str | int]],
    min_count_request_for_consideration: int = 10,
    iteration: int = 0,
) -> tuple[defaultdict[str, defaultdict[str, int]], list[list[str | int]]]:
    """Iterate the alignment process.

    Args:
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        dictionary: Dictionary to align
        corpus_path: Path to the corpus to align
        fast_align_path: Path to the fast align binary
        token_mapper: Token mapper
        alignment_mode: Alignment mode
        alignment_log: Alignment log
        min_count_request_for_consideration: Minimum count request for consideration
        iteration: Iteration number

    Returns:
        Token mapper and alignment log
    """
    logger.info("====================================================")
    logger.info(f"Iteration {iteration}")
    logger.info(
        f"Min count request for consideration: {min_count_request_for_consideration}"
    )

    # Create a new instance
    new_token_mapper = TokenMpper()
    new_token_mapper.tokenized_possible_translations.update(
        token_mapper.tokenized_possible_translations
    )

    output_corpus_path = (
        corpus_path.split(".")[0] + f".iteration_{iteration}.loop_0.moses"
    )
    deleted_target_tokenizer, new_token_mapper, alignment_log = align_tokens(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        dictionary=dictionary,
        corpus_path=output_corpus_path,
        fast_align_path=fast_align_path,
        token_mapper=new_token_mapper,
        alignment_mode=alignment_mode,
        alignment_log=alignment_log,
        min_count_request_for_consideration=min_count_request_for_consideration,
        iteration=iteration,
    )
    # If we have reached the minimum count, we can stop the alignment process
    min_count_request_for_consideration = min_count_request_for_consideration + 1
    if min_count_request_for_consideration > 2:
        ## Map special tokens
        new_token_mapper.map_special_tokens(
            new_tokenizer=target_tokenizer, old_tokenizer=source_tokenizer
        )

        logger.info(f"Alignment completed in {iteration} iterations.")
        logger.info(
            f"{len(new_token_mapper.tokenized_possible_translations)} tokens were mapped."
        )
        return new_token_mapper.tokenized_possible_translations, alignment_log
    else:
        iteration += 1
        return iterate_alignment(
            source_tokenizer=source_tokenizer,
            target_tokenizer=deleted_target_tokenizer,
            dictionary=dictionary,
            corpus_path=corpus_path,
            fast_align_path=fast_align_path,
            token_mapper=new_token_mapper,
            alignment_mode=alignment_mode,
            alignment_log=alignment_log,
            min_count_request_for_consideration=min_count_request_for_consideration,
            iteration=iteration,
        )


def probabilities_for_token(
    translations: dict[str, int], unk_token: str
) -> list[tuple[str, float]]:
    """Calculate the probability of each translation for a token.
    Args:
        translations: Translations for a token
        unk_token: UNK Token for an invalid mapping.

    Returns:
        Probabilities for each translation
    """
    total = sum(translations.values())
    if total == 0:
        logger.warning(f"Token has no valid translations: {translations}")
        return [(unk_token, 1.0)]
    return [(translation, count / total) for translation, count in translations.items()]


def token_alignment_probabilities(
    target_tokenizer: PreTrainedTokenizerFast,
    tokenized_possible_translations: dict[str, dict[str, int]],
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Compute the probability of each translation for each token.

    Args:
        target_tokenizer: Target tokenizer
        tokenized_possible_translations: Tokenized possible translations

    Returns:
        Mapping of tokens to their possible translations
    """
    token_mappings: list[tuple[str, list[tuple[str, float]]]] = []
    target_vocab = target_tokenizer.get_vocab()
    translated_token_count = 0
    unk_token = target_tokenizer.unk_token
    if unk_token is None:
        unk_token = "[UNK]"
    if isinstance(unk_token, list):
        unk_token = unk_token[0]
    for token in target_vocab:
        if (
            token not in tokenized_possible_translations
            or sum(tokenized_possible_translations[token].values()) == 0
        ):
            print(f"Token not mapped: {token}")
            token_mappings.append((token, [(unk_token, 1.0)]))
            continue
        translated_token_count += 1
        translations = tokenized_possible_translations[token]
        token_mappings.append((token, probabilities_for_token(translations, unk_token)))
    target_vocab_size = len(target_vocab)
    readable_ratio = translated_token_count / target_vocab_size
    logger.info(f"Target vocab size: {target_vocab_size}")
    logger.info(f"Total translated tokens: {translated_token_count}")
    logger.info(
        f"Percentage of translated tokens: {readable_ratio:.2f} ({readable_ratio * 100:.2f}%)"
    )
    print(f"Percentage of translated tokens: {readable_ratio * 100:.2f}%")
    return token_mappings


@dataclass
class RecurrentAlignmentArgs:
    source_model: str
    target_tokenizer: str
    dictionary: str
    corpus_path: str
    fast_align_path: str
    alignment_mode: AlignmentMode
    mapping_mode: str
    min_count: int
    output_dir: str
    logging: str
    log_file: Path | None = None


def parse_args() -> RecurrentAlignmentArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=str, required=True)
    parser.add_argument("--target-tokenizer", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--fast-align-path", type=str, required=True)
    parser.add_argument("--alignment-mode", type=str, required=True)
    parser.add_argument("--mapping-mode", type=str, required=True)
    parser.add_argument("--min-count", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--logging", type=str, default="INFO")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()
    return RecurrentAlignmentArgs(**vars(args))


def run_recurrent_alignment(
    source_model: str,
    target_tokenizer: str,
    dictionary: str,
    corpus_path: str,
    fast_align_path: str,
    alignment_mode: str,
    mapping_mode: str,
    min_count: int,
    output_dir: str,
    logging_level: str = "INFO",
    log_file: Path | None = None,
    seed: int = 42,
) -> None:
    """Run the recurrent alignment process.

    Args:
        source_model: Path to the source model
        target_tokenizer: Path to the target tokenizer
        dictionary: Path to the bilingual dictionary
        corpus_path: Path to the corpus
        fast_align_path: Path to the fast align binary
        alignment_mode: Alignment mode ('word' or 'token')
        mapping_mode: Mapping mode ('replace')
        min_count: Minimum count for consideration
        output_dir: Output directory
        logging_level: Logging level
        log_file: Log file path
        seed: Random seed
    """
    # Setup logging
    setup_logging(
        log_level=logging_level,
        log_file=log_file,
    )

    logger.info("Starting token alignment process")
    logger.info(f"Source model: {source_model}")
    logger.info(f"Target tokenizer: {target_tokenizer}")
    logger.info(f"Alignment mode: {alignment_mode}")
    logger.info(f"Mapping mode: {mapping_mode}")

    set_seed(seed)

    corpus = corpus_path
    os.makedirs(os.path.dirname(corpus), exist_ok=True)

    alignment_mode_enum = (
        AlignmentMode.WORD if alignment_mode == "word" else AlignmentMode.TOKEN
    )
    min_count_request_for_consideration = min_count

    export_dir = output_dir
    os.makedirs(export_dir, exist_ok=True)

    source_tokenizer_obj = AutoTokenizer.from_pretrained(
        source_model, add_prefix_space=True
    )
    target_tokenizer_obj = AutoTokenizer.from_pretrained(
        target_tokenizer, add_prefix_space=True
    )

    dictionary_obj = load_bilingual_dict(dictionary)

    token_mapper = TokenMpper()
    alignment_log: list[list[str | int]] = [
        ["Iteration", "loop", "Mapped Tokens", "All Mapped Tokens", "Min Count"]
    ]
    tokenized_possible_translations, completed_alignment_log = iterate_alignment(
        source_tokenizer_obj,
        target_tokenizer_obj,
        dictionary_obj,
        corpus,
        fast_align_path,
        token_mapper,
        alignment_mode_enum,
        alignment_log,
        min_count_request_for_consideration,
    )

    # Save the alignment log
    completed_alignment_log_path = os.path.join(export_dir, "alingment_log.csv")
    with open(completed_alignment_log_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(completed_alignment_log)

    # Remap the model
    mapping = token_alignment_probabilities(
        target_tokenizer_obj, dict(tokenized_possible_translations)
    )

    if mapping_mode == "replace":
        model = remap_model(source_model, target_tokenizer, mapping, source_model)

        # Save the remapped model
        new_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)
        model.save_pretrained(export_dir)
        new_tokenizer.save_pretrained(export_dir)
    else:
        raise ValueError("Invalid mapping mode.")
