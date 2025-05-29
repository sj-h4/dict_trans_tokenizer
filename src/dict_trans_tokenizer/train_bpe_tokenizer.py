import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from dict_trans_tokenizer.utils.load_bilingual_dict import load_bilingual_dict


def train_bpe_tokenizer(
    train_words: list[str],
    output_dir: str,
    vocab_size: int,
    show_progress: bool = False,
    special_tokens: list[str] | None = None,
) -> None:
    """Train a BPE tokenizer from a list of words.
    
    Args:
        train_words: List of words to train the tokenizer on
        output_dir: Directory to save the tokenizer
        vocab_size: Vocabulary size for the tokenizer
        show_progress: Whether to show training progress
        special_tokens: List of special tokens, defaults to standard set
    """
    if special_tokens is None:
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=show_progress,
        special_tokens=special_tokens,
    )

    tokenizer.train_from_iterator(iterator=train_words, trainer=trainer)

    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save(f"{output_dir}/tokenizer.json")

    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )
    tokenizer_fast.save_pretrained(f"{output_dir}/tokenizer-fast")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_list_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--vocab_size", type=int, default=1000)
    args = parser.parse_args()

    vocab_size = args.vocab_size

    vocab_list_path = args.vocab_list_path
    vocab_dict = load_bilingual_dict(vocab_list_path)
    train_words = [entry.entry.lower().strip() for entry in vocab_dict]
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists.")
        exit(0)

    train_bpe_tokenizer(
        train_words, output_dir, vocab_size=vocab_size, show_progress=True
    )
