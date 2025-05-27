"""
Basic usage example of dict-trans-tokenizer package.

This example demonstrates the complete workflow:
1. Train a BPE tokenizer from dictionary data
2. Create an aligned corpus
3. Run recurrent alignment using the dummy Manchu-English dictionary
"""

import tempfile
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from dict_trans_tokenizer import (
    AlignmentMode,
    create_aligned_corpus,
    load_bilingual_dict,
    run_recurrent_alignment,
    train_bpe_tokenizer,
)


def main():
    """Run a complete example using the dummy Manchu-English dictionary."""
    source_model_name = "roberta-base"

    # Paths
    current_dir = Path(__file__).parent
    dictionary_path = current_dir / "data" / "mnc-en.json"
    temp_dir = Path(tempfile.mkdtemp())

    print("=== Dict Trans Tokenizer Example ===")
    print(f"Using dictionary: {dictionary_path}")
    print(f"Working directory: {temp_dir}")

    try:
        # Step 1: Train target (Manchu) tokenizer
        print("\n1. Training Manchu BPE tokenizer...")
        vocab_dict = load_bilingual_dict(str(dictionary_path))

        # Extract Manchu words for tokenizer training
        manchu_words = [entry.entry.lower().strip() for entry in vocab_dict]
        print(f"Found {len(manchu_words)} Manchu words: {manchu_words}")

        manchu_tokenizer_dir = temp_dir / "manchu_tokenizer"
        train_bpe_tokenizer(
            train_words=manchu_words * 10,  # Repeat to have enough data
            output_dir=str(manchu_tokenizer_dir),
            vocab_size=50,  # Small vocab for demo
        )
        print(f"Manchu tokenizer saved to: {manchu_tokenizer_dir}")

        # Step 2: Create aligned corpus
        print("\n2. Creating aligned corpus...")
        source_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            source_model_name
        )
        target_tokenizer: PreTrainedTokenizerFast = (
            PreTrainedTokenizerFast.from_pretrained(
                str(manchu_tokenizer_dir / "tokenizer-fast")
            )
        )
        create_aligned_corpus(
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            dictionary=vocab_dict,
            output_path=str(temp_dir / "corpus.moses"),
            alignment_mode=AlignmentMode.TOKEN,
        )

        # Step 3: Run recurrent alignment
        print("\n3. Running recurrent alignment...")
        corpus_path = str(temp_dir / "corpus.moses")
        output_dir = str(temp_dir / "aligned_output")

        # Note: This will fail without fast_align, but shows the complete workflow
        try:
            run_recurrent_alignment(
                source_model=source_model_name,
                target_tokenizer=str(manchu_tokenizer_dir / "tokenizer-fast"),
                dictionary=str(dictionary_path),
                corpus_path=corpus_path,
                fast_align_path="fast_align",  # Assumes fast_align is in PATH
                alignment_mode="token",
                mapping_mode="replace",
                min_count=0,
                output_dir=output_dir,
                logging_level="INFO",
                seed=42,
            )

            print(f"\nAlignment completed! Results saved to: {output_dir}")

        except Exception as e:
            print(f"\nAlignment failed (expected if fast_align not installed): {e}")
            print("\nTo complete this example, install the forked fast_align:")
            print("  git clone https://github.com/FremyCompany/fast_align.git")

        print(f"\nTemporary files in: {temp_dir}")
        print("You can examine the tokenizer files and training data there.")

    except Exception as e:
        print(f"Error in example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

