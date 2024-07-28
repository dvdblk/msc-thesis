def fix_bert_tokenization(tokens):
    """Fixes BERT style tokenization so that it can be used for visualization with pdf_heatmap"""
    fixed_tokens = []
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            fixed_tokens.append(token[2:])
        elif token in ["", ",", ".", ":", ";", "!", "?"]:
            if i > 0 and not tokens[i - 1] in ["", ",", ".", ":", ";", "!", "?"]:
                fixed_tokens.append(token + " ")
            else:
                fixed_tokens.append(token)
        elif (
            i < len(tokens) - 1
            and not tokens[i + 1].startswith("##")
            and tokens[i + 1] not in ["", ",", ".", ":", ";", "!", "?"]
        ):
            fixed_tokens.append(token + " ")
        else:
            fixed_tokens.append(token)

    # Remove trailing space from the last token if it exists
    if fixed_tokens and fixed_tokens[-1].endswith(" "):
        fixed_tokens[-1] = fixed_tokens[-1].rstrip()

    return fixed_tokens
