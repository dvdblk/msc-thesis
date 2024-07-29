def fix_bert_tokenization(tokens):
    """Fixes BERT tokenization by removing special tokens and merging subwords
    for easier SDGRS visualization that resembles SHAP style merged tokens.
    """
    fixed_tokens = []
    for i, token in enumerate(tokens):
        if i == 0 and token == "[CLS]":
            fixed_tokens.append(token)
        elif i == len(tokens) - 1 and token == "[SEP]":
            fixed_tokens.append(token)
        elif token.startswith("##"):
            fixed_tokens.append(token[2:])
            if (
                i < len(tokens) - 1
                and tokens[i + 1] not in ["[SEP]", "", ",", ".", ":", ";", "!", "?"]
                and not tokens[i + 1].startswith("##")
            ):
                fixed_tokens[-1] += " "
        elif token in ["", ",", ".", ":", ";", "!", "?"]:
            if i > 0 and not tokens[i - 1] in [
                "[CLS]",
                "",
                ",",
                ".",
                ":",
                ";",
                "!",
                "?",
            ]:
                fixed_tokens.append(token + " ")
            else:
                fixed_tokens.append(token)
        elif (
            i < len(tokens) - 1
            and not tokens[i + 1].startswith("##")
            and tokens[i + 1] not in ["[SEP]", "", ",", ".", ":", ";", "!", "?"]
        ):
            fixed_tokens.append(token + " ")
        else:
            fixed_tokens.append(token)

    # Remove trailing space from the last token if it exists and it's not [SEP]
    if fixed_tokens and fixed_tokens[-1].endswith(" ") and fixed_tokens[-1] != "[SEP]":
        fixed_tokens[-1] = fixed_tokens[-1].rstrip()

    return fixed_tokens


def prepare_fixed_bert_tokens_for_pdf_viz(tokens):
    """
    Move space from end of word to beginning of next one as an underscore
    """
    for i in range(1, len(tokens)):
        if tokens[i - 1].endswith(" ") and tokens[i] != "[SEP]":
            tokens[i] = "▁" + tokens[i]
            tokens[i - 1] = tokens[i - 1].rstrip()

    return tokens
