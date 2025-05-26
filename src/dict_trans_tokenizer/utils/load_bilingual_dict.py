import json

from dict_trans_tokenizer.types.bilingual_dict import BilingualDict


def load_bilingual_dict(dict_path: str) -> list[BilingualDict]:
    """Load a bilingual dictionary

    Args:
        dict_path: a dictionary path

    Returns:
        A list of `BilingualDict`
    """
    with open(dict_path) as f:
        dict_json = json.load(f)
    bilingual_dict = [BilingualDict(**entry) for entry in dict_json]
    return bilingual_dict
