import altair as alt
import polars as pl
from camminapy.plot import altair_theme

altair_theme()
alt.data_transformers.disable_max_rows()


def get_df() -> pl.DataFrame:
    def bundeslaender(code):
        d = {
            "TH": "Thüringen",
            "HB": "Bremen",
            "SH": "Schleswig-Holstein",
            "BW": "Baden-Württemberg",
            "SN": "Sachsen",
            "MV": "Mecklenburg-Vorpommern",
            "BB": "Brandenburg",
            "BY": "Bayern",
            "BE": "Berlin",
            "ST": "Sachsen-Anhalt",
            "NW": "Nordrhein-Westfalen",
            "NI": "Niedersachsen",
            "HE": "Hessen",
            "SL": "Saarland",
            "HH": "Hamburg",
            "RP": "Rheinland-Pfalz",
        }

        return d[code]

    df = pl.read_csv("data/Abiturnoten.csv").with_columns(
        pl.col("Bundesland").apply(lambda code: bundeslaender(code))
    )

    df_deutschland = (
        df.groupby("Jahr", "Note", maintain_order=True)
        .agg(pl.col("Anzahl").sum(), pl.lit("Gesamt").alias("Bundesland"))
        .with_columns(
            (100 * pl.col("Anzahl") / pl.col("Anzahl").sum().over("Jahr", "Bundesland"))
            .round(1)
            .alias("Anteil in Prozent")
        )
    )

    df = pl.concat(
        [
            df,
            df_deutschland.select(
                "Jahr", "Note", "Bundesland", "Anzahl", "Anteil in Prozent"
            ),
        ]
    ).with_columns(
        pl.when(pl.col("Bundesland") == "Gesamt")
        .then("Deutschland")
        .otherwise("Bundesländer")
        .alias("Gruppierung")
    )
    return df


def get_chart_time_series(df):
    chart = (
        alt.Chart(
            df.with_columns(
                pl.when(pl.col("Gruppierung") == "Deutschland")
                .then("Germany")
                .otherwise("Individual states")
                .alias("Gruppierung")
            ),
        )
        .mark_line(clip=True, point=False, size=4)
        .encode(
            x=alt.X("Jahr:Q")
            .scale(domain=(2005.6, 2022.5))
            .axis(format="d")
            .title("Year"),
            y=alt.Y("Anteil in Prozent:Q").scale(zero=True).title("Share in %"),
            color=alt.Color("Gruppierung:N")
            .scale(range=["blue", "gray"])
            .title("Grouping"),
            opacity=alt.condition(
                alt.datum["Gruppierung"] == "Germany", alt.value(1.0), alt.value(0.2)
            ),
            detail="Bundesland:N",
        )
        .transform_filter(alt.datum["Note"] == 1.0)
        .properties(width=1300, height=700)
        .properties(
            title={
                # "text": "High-school diploma with top grade (1.0)",
                "text": "COVID-19 in Germany: share of high-school diplomas with top grade up by 78%",
                **{
                    "subtitle": [
                        "Numbers up by 78.9% when comparing the years 2019 and 2022. Top grade refers to an Abitur with grade 1.0.",
                        "Data: https://www.kmk.org/dokumentation-statistik/statistik/schulstatistik/abiturnoten.html",
                        "Analysis and visualization: Thomas Camminady",
                    ],
                    "subtitleFontSize": 8,
                    "subtitleFontWeight": "lighter",
                    "subtitleColor": "gray",
                    "anchor": "middle",
                },
            },
        )
    )

    (
        alt.Chart(pl.DataFrame({"from": [2006, 2019.5], "to": [2019.5, 2022]}))
        .mark_area(
            line={"color": "white"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="white", offset=0),
                    alt.GradientStop(color="white", offset=1),
                ],
                x1=0,
                x2=1,
                y1=1,
                y2=1,
            ),
        )
        .encode(
            x=alt.X("from:Q").title("Jahr"),
            x2=alt.X2("to:Q"),
            y=alt.value(0.0),
            y2=alt.value(700),
            opacity=alt.value(0.1),
        )
    )
    (
        alt.Chart(pl.DataFrame({"from": [2019.5, 2022], "to": [2022, 2022]}))
        .mark_area(
            line={"color": "white"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="white", offset=0),
                    alt.GradientStop(color="black", offset=1),
                ],
                x1=0,
                x2=1,
                y1=1,
                y2=1,
            ),
        )
        .encode(
            x=alt.X("from:Q").title("Jahr"),
            x2=alt.X2("to:Q"),
            y=alt.value(0.0),
            y2=alt.value(700),
            opacity=alt.value(0.1),
        )
    )
    (
        alt.Chart(
            pl.DataFrame(
                {
                    "Jahr": [2021.9],
                    "Anteil in Prozent": [0.2],
                    "Text": ["COVID-19 Pandemie"],
                }
            )
        )
        .mark_text(fontSize=18, align="right")
        .encode(x="Jahr:Q", y="Anteil in Prozent:Q", text="Text:N")
    )
    # background_pre + background_post + chart + text1
    return (
        chart
        + chart.mark_text(dx=-7, dy=-13, fontSize=14, fontWeight="bold", clip=True)
        .encode(text="Anteil in Prozent:N")
        .transform_filter(alt.datum["Gruppierung"] == "Germany")
        + chart.mark_point(size=100, filled=True, clip=True).transform_filter(
            alt.datum["Gruppierung"] == "Germany"
        )
    )


def get_chart_grid(df):
    def get_chart(grade_min, grade_max, cmap, text, reverse=False):
        return (
            alt.Chart(
                df.filter(pl.col("Note").is_between(grade_min, grade_max))
                .with_columns((pl.col("Jahr") % 100).apply(lambda s: f"'{s:02d}"))
                .with_columns(pl.col("Note").cast(str)),
            )
            .mark_area()
            .encode(
                x=alt.X("Jahr:N")
                .title("Year")
                .axis(
                    values=df.with_columns(
                        (pl.col("Jahr") % 100).apply(lambda s: f"'{s:02d}")
                    )["Jahr"]
                    .sort()
                    .unique(maintain_order=True)
                    .to_list()[::2]
                ),
                y=alt.Y("Anteil in Prozent:Q").title("Share in %"),
                color=alt.Color("Note:N")
                .scale(scheme=cmap, reverse=reverse)
                .title("Grade"),
                facet=alt.Facet(
                    "Bundesland:N",
                    columns=4,
                    spacing=20,
                ).title(None),
            )
            .properties(width=250, height=200)
            .transform_filter(alt.datum["Bundesland"] != "Gesamt")
            .properties(
                title={
                    "text": text,
                    "fontSize": 30,
                },
            )
        )

    return (
        alt.vconcat(
            get_chart(
                1.0, 1.9, "lighttealblue", "Share of good grades rises", reverse=True
            ),
            get_chart(
                2.0, 2.9, "warmgreys", "Share of average grades plateaus", reverse=True
            ),
            get_chart(
                3.0, 4.0, "lightorange", "Share of bad grades is down", reverse=False
            ),
        )
        .resolve_scale(color="independent")
        .properties(
            title={
                # "text": "High-school diploma with top grade (1.0)",
                "text": "Grade inflation for the German Abitur",
                "fontSize": 60,
                "subtitle": [
                    "Data: https://www.kmk.org/dokumentation-statistik/statistik/schulstatistik/abiturnoten.html - "
                    "Analysis and visualization: Thomas Camminady",
                ],
                "subtitleFontSize": 8,
                "subtitleFontWeight": "lighter",
                "subtitleColor": "gray",
                "anchor": "middle",
            },
        )
    )


def run():
    df = get_df()
    chart = get_chart_time_series(df)
    chart.save("timeseries.png", scale_factor=3)
    chart = get_chart_grid(df)
    chart.save("grid.png", scale_factor=3)
