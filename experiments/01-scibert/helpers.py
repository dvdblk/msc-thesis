from typing import Optional

import pandas as pd


def read_df(
    filepath: str,
    coltypes: dict,
    keep_default_na: Optional[bool] = True,
    skiprows: Optional[list[int]] = None,
    new_names: Optional[list[str]] = None,
    index_col: Optional[int] = None,
) -> pd.DataFrame:

    df = pd.read_csv(
        filepath,
        keep_default_na=keep_default_na,
        skiprows=skiprows,
        names=new_names,
        index_col=index_col,
        sep="\t",
        encoding="utf-8",
    )

    df = df.astype(coltypes)

    return df


def write_df(
    df,
    filename,
    index: bool,
    na_rep: Optional[str] = None,
    header: Optional[bool] = True,
) -> None:
    df.to_csv(
        filename, sep="\t", index=index, na_rep=na_rep, header=header, encoding="utf-8"
    )
