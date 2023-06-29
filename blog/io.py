from abc import ABC, abstractmethod
from typing import Callable

import polars as pl

from blog import logger


class Piper(ABC):
    def __init__(self, to_log: bool = True):
        self.to_log = to_log

    @property
    def pipe(self) -> Callable[[pl.DataFrame], pl.DataFrame]:
        """Returns the pipe wrapped in logging information."""
        return self._wrapped_pipe

    def _wrapped_pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        name = type(self).__name__
        initial_shape = df.shape
        df = self._pipe(df)
        if self.to_log:
            logger.info(
                f"Calling {name} changed df.shape from {initial_shape} to {df.shape}"
            )
        return df

    @abstractmethod
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        pass


class PiperFilterGermany(Piper):
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            pl.col("countries_en").str.to_lowercase().str.contains("germany")
        )


class PiperValidCode(Piper):
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.filter(pl.col("code").is_null().is_not())
        duplicate_codes = (
            df.groupby("code")
            .count()
            .sort("count")
            .filter(pl.col("count") > 1)["code"]
            .to_list()
        )
        df = df.filter(pl.col("code").is_in(duplicate_codes).is_not())
        return df


class PiperConvertNutrientsToFloats(Piper):
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        # Convert ingredients from string to float
        ingredient_columns = [c for c in df.columns if "_100g" in c]
        return df.with_columns(pl.col(c).cast(float) for c in ingredient_columns)


class PiperHasNutriScore(Piper):
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("nutriscore_grade").is_null().is_not())


class PiperConvertDatetimeStrings(Piper):
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("created_datetime").str.strptime(pl.Datetime),
                pl.col("last_modified_datetime").str.strptime(pl.Datetime),
            )
            .sort("created_datetime")
            .drop("created_t", "last_modified_t")
        )


class PiperKeepOnlyEnglishVersion(Piper):
    # some columns exist multiple times
    # and we only need to keep the version with _en
    # because that is the one formatted nicely
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        en_columns = [c for c in df.columns if c.endswith("_en")]
        en_columns_without_en = [c.replace("_en", "") for c in en_columns]
        columns_to_remove = [
            c
            for c in df.columns
            if c.replace("_en", "") in en_columns_without_en and not c.endswith("_en")
        ] + [
            c
            for c in df.columns
            if c.split("_")[0] in en_columns_without_en and not c.endswith("_en")
        ]
        return df.drop(columns=columns_to_remove)


class PiperKeepOnlyTagsVersion(Piper):
    # some columns exist multiple times
    # and we only need to keep the version with _tags
    # because that is the one formatted nicely
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        tags_columns = [c for c in df.columns if c.endswith("_tags")]
        tags_columns_without_en = [c.replace("_tags", "") for c in tags_columns]
        columns_to_remove = [
            c
            for c in df.columns
            if c.replace("_tags", "") in tags_columns_without_en
            and not c.endswith("_tags")
        ] + [
            c
            for c in df.columns
            if c.split("_")[0] in tags_columns_without_en and not c.endswith("_tags")
        ]
        return df.drop(columns=columns_to_remove)


class PiperConvertNutriScoresToCapitalLetters(Piper):
    def _pipe(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("nutriscore_grade").str.to_uppercase())


ConvertNutrientsToFloats = PiperConvertNutrientsToFloats().pipe
ConvertDatetimeStrings = PiperConvertDatetimeStrings().pipe
KeepOnlyEnglishVersion = PiperKeepOnlyEnglishVersion().pipe
ValidCode = PiperValidCode().pipe
FilterGermany = PiperFilterGermany().pipe
HasNutriScore = PiperHasNutriScore().pipe
KeepOnlyTagsVersion = PiperKeepOnlyTagsVersion().pipe
ConvertNutriScoresToCapitalLetters = PiperConvertNutriScoresToCapitalLetters().pipe
