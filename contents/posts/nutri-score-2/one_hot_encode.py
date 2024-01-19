import collections

import polars as pl

from blog import logger


def one_hot_encode(
    df: pl.DataFrame,
    column: str,
    n: int = 20,
    remove_prefix: str | list[str] | None = None,
    dtype=int,
) -> pl.DataFrame:
    lol_categories = (
        df.select(column).with_columns(pl.col(column).str.split(","))[column].to_list()
    )

    categories = []
    for _l in lol_categories:
        if _l is not None:
            categories.extend(_l)

    hist = dict(collections.Counter(categories))

    df_hist = (
        pl.DataFrame(
            {
                column: [key for key, _ in hist.items()],
                "count": [value for _, value in hist.items()],
            }
        )
        .sort("count", descending=True)
        .head(n)
    )

    df = (
        df.with_columns(
            *[
                (pl.col(column).str.to_lowercase().str.contains(x.lower()))
                .cast(dtype)
                .alias(f"{x}")
                for x in df_hist[column].to_list()
            ],
        )
        .select("code", *[f"{x}" for x in df_hist[column].to_list()])
        .sort("code")
    )
    if isinstance(remove_prefix, str):
        df = df.rename(
            mapping={x: x.replace(remove_prefix, "") for x in df_hist[column].to_list()}
        )
    if isinstance(remove_prefix, list):
        for s in remove_prefix:
            df = df.rename(
                mapping={x: x.replace(s, "") for x in df_hist[column].to_list()}
            )
    logger.info(df.shape)
    return df.fill_null(0)
