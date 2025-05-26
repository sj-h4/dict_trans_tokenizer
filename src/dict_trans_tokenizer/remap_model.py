import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


def remap_model(
    source_tokenizer: str,
    target_tokenizer: str,
    mapping: list[tuple[str, list[tuple[str, float]]]],
    source_model: str,
) -> PreTrainedModel:
    """Remap the embeddings of a model trained with one tokenizer to another tokenizer.
    Derived from <https://github.com/LAGoM-NLP/transtokenizer/blob/99a41391da7b6045e80cc6f49bf7cf23abca9056/transtokenizers/transtokenizers.py>

    Args:
        source_tokenizer: [TODO:description]
        target_tokenizer: [TODO:description]
        mapping: [TODO:description]
        source_model: [TODO:description]

    Returns:
        Remapped model
    """
    # load tokenizers for the two models
    old_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer)
    new_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)

    # add an unk token if none exist
    if old_tokenizer.unk_token_id is None:
        if old_tokenizer.pad_token_id is not None:
            old_tokenizer.unk_token_id = old_tokenizer.pad_token_id
            old_tokenizer.unk_token = old_tokenizer.pad_token
        elif old_tokenizer.bos_token_id is not None:
            old_tokenizer.unk_token_id = old_tokenizer.bos_token_id
            old_tokenizer.unk_token = old_tokenizer.bos_token
        else:
            print(
                "WARNING: The old tokenizer had neither UNK, PAD or BOS special tokens"
            )
            old_tokenizer.unk_token_id = 0

    # load the old model
    print("Loading the source model...")
    model = AutoModelForCausalLM.from_pretrained(source_model)

    # remap the embeddings
    print("Remapping the model...")
    with torch.no_grad():
        # get the embeddings of the OLM model
        old_embeddings = model.get_input_embeddings().weight.data
        old_output_embeddings = model.get_output_embeddings().weight.data
        tied_weights = model.config.tie_word_embeddings

        # change the tokenizer of the OLM model to the one of the RobBERT model, and reinitialize the embeddings
        # model.resize_token_embeddings(
        #     1
        # )  # this is a hack to make the model forget its old tokenizer
        # HACK: Use the same distribution for the new embeddings as the old ones
        model.resize_token_embeddings(
            len(new_tokenizer), mean_resizing=False
        )  # this is the actual call to change the tokenizer
        new_embeddings = model.get_input_embeddings()
        new_output_embeddings = model.get_output_embeddings()
        model.config.vocab_size = len(new_tokenizer)
        model.config.pad_token_id = new_tokenizer.pad_token_id
        model.config.bos_token_id = new_tokenizer.bos_token_id
        model.config.eos_token_id = new_tokenizer.eos_token_id
        model.config.unk_token_id = new_tokenizer.unk_token_id
        model.config.sep_token_id = new_tokenizer.sep_token_id
        model.config.cls_token_id = new_tokenizer.cls_token_id
        model.config.mask_token_id = new_tokenizer.mask_token_id
        model.config.additional_special_tokens_ids = (
            new_tokenizer.additional_special_tokens_ids
        )
        model.config.tokenizer_class = new_tokenizer.__class__.__name__

        # debug info
        # print(old_embeddings.shape)
        # print(old_output_embeddings.shape)
        # print(new_embeddings.weight.data.shape)
        # print(new_output_embeddings.weight.data.shape)

        # for each token in the new tokenizer, find the corresponding tokens in the old tokenizer, and average their embeddings
        from functools import reduce

        from tqdm import tqdm

        for new_id in tqdm(range(len(new_tokenizer))):
            # new_token = new_tokenizer.convert_ids_to_tokens(new_id)
            old_tokens = mapping[new_id][1]  # list of (ids,weight) in the old tokenizer

            # sort the list such that the smallest weights come first (for numerical stability)
            old_tokens = sorted(old_tokens, key=lambda x: x[1])

            # map tokens to their ids
            old_ids = [
                (old_tokenizer.convert_tokens_to_ids(old_token), weight)
                for old_token, weight in old_tokens
            ]
            old_ids = [
                (old_id if old_id is not None else old_tokenizer.unk_token_id, weight)
                for old_id, weight in old_ids
            ]

            # we use a weighted average, where the first token in the list has 0.4 weight, the second 0.2, and the remaining 0.4 are equally distributed among all tokens (including the first two)
            if len(old_ids) > 1:
                new_embeddings.weight.data[new_id] = reduce(
                    lambda a, b: a.add_(b),
                    [old_embeddings[old_id] * weight for old_id, weight in old_ids],
                )
                if not (tied_weights):
                    new_output_embeddings.weight.data[new_id] = reduce(
                        lambda a, b: a.add_(b),
                        [
                            old_output_embeddings[old_id] * weight
                            for old_id, weight in old_ids
                        ],
                    )
            elif len(old_ids) == 1:
                new_embeddings.weight.data[new_id] = old_embeddings[old_ids[0][0]]
                if not (tied_weights):
                    new_output_embeddings.weight.data[new_id] = old_output_embeddings[
                        old_ids[0][0]
                    ]
            else:  # use the unknown token embedding if the token is not in the old tokenizer
                new_embeddings.weight.data[new_id] = old_embeddings[
                    old_tokenizer.unk_token_id
                ]
                if not (tied_weights):
                    new_output_embeddings.weight.data[new_id] = old_output_embeddings[
                        old_tokenizer.unk_token_id
                    ]

    return model
