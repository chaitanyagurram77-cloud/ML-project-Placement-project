"""
ml/eda.py
---------
Exploratory Data Analysis helpers.

Provides:
- Dataset shape
- Column names
- Missing values
- Numeric summary statistics
- Categorical summary
- Placement rate
- Correlation with target
- EDA charts
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.data_loader import NUMERIC_COLS, TARGET_COL


sns.set_theme(style="whitegrid")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHARTS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)


# ============================================================
# CREATE CHART DIRECTORY
# ============================================================

def _ensure_charts_dir():
    os.makedirs(
        CHARTS_DIR,
        exist_ok=True
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Generate summary information for the dataset.
    """

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_cols = [
        column
        for column in NUMERIC_COLS
        if column in df.columns
    ]

    # --------------------------------------------------------
    # Describe numeric columns
    # --------------------------------------------------------

    if numeric_cols:

        describe_df = (
            df[numeric_cols]
            .describe()
            .T
            .round(2)
        )

        numeric_describe = describe_df.to_dict(
            orient="index"
        )

    else:

        numeric_describe = {}

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_cols = (
        df.select_dtypes(
            include="object"
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # Categorical summary
    # --------------------------------------------------------

    categorical_summary = {}

    for column in categorical_cols:

        categorical_summary[column] = (
            df[column]
            .value_counts()
            .head(10)
            .to_dict()
        )

    # --------------------------------------------------------
    # Placement rate
    # --------------------------------------------------------

    placement_rate = None

    if TARGET_COL in df.columns:

        placement_rate = round(
            df[TARGET_COL].mean() * 100,
            2
        )

    # --------------------------------------------------------
    # Return summary
    # --------------------------------------------------------

    return {

        # Used by app.py
        "shape": df.shape,

        "columns": df.columns.tolist(),

        "missing_values": (
            df.isnull()
            .sum()
            .to_dict()
        ),

        # Additional EDA information
        "numeric_describe": numeric_describe,

        "categorical_summary": categorical_summary,

        "placement_rate": placement_rate,

        "correlation_with_target":
            _correlation_with_target(
                df,
                numeric_cols
            )
    }


# ============================================================
# CORRELATION WITH TARGET
# ============================================================

def _correlation_with_target(
    df: pd.DataFrame,
    numeric_cols
):

    if TARGET_COL not in df.columns:
        return {}

    # Keep only columns that exist
    available_cols = [
        column
        for column in numeric_cols
        if column in df.columns
    ]

    # Add target
    if TARGET_COL not in available_cols:
        available_cols.append(TARGET_COL)

    # Need at least two columns
    if len(available_cols) < 2:
        return {}

    corr = (
        df[available_cols]
        .corr()[TARGET_COL]
        .drop(TARGET_COL)
    )

    corr = (
        corr
        .sort_values(
            ascending=False
        )
        .round(3)
    )

    return corr.to_dict()


# ============================================================
# GENERATE EDA CHARTS
# ============================================================

def generate_charts(df: pd.DataFrame) -> list:
    """
    Generate EDA charts and save them in:

        static/charts/

    Returns a list of generated filenames.
    """

    _ensure_charts_dir()

    filenames = []


    # ========================================================
    # 1. PLACEMENT DISTRIBUTION
    # ========================================================

    if TARGET_COL in df.columns:

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        sns.countplot(
            x=TARGET_COL,
            data=df,
            hue=TARGET_COL,
            ax=ax,
            palette="viridis",
            legend=False
        )

        ax.set_title(
            "Placement Status Distribution"
        )

        ax.set_xlabel(
            "Placement Status (0 = Not Placed, 1 = Placed)"
        )

        ax.set_ylabel(
            "Number of Students"
        )

        fname = (
            "placement_distribution.png"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                CHARTS_DIR,
                fname
            ),
            dpi=110
        )

        plt.close(fig)

        filenames.append(fname)


    # ========================================================
    # 2. CGPA DISTRIBUTION
    # ========================================================

    if (
        "CGPA" in df.columns
        and TARGET_COL in df.columns
    ):

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        sns.histplot(
            data=df,
            x="CGPA",
            hue=TARGET_COL,
            kde=True,
            ax=ax,
            palette="viridis",
            element="step"
        )

        ax.set_title(
            "CGPA Distribution by Placement Status"
        )

        ax.set_xlabel(
            "CGPA"
        )

        ax.set_ylabel(
            "Number of Students"
        )

        fname = (
            "cgpa_distribution.png"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                CHARTS_DIR,
                fname
            ),
            dpi=110
        )

        plt.close(fig)

        filenames.append(fname)


    # ========================================================
    # 3. CORRELATION HEATMAP
    # ========================================================

    numeric_cols = [
        column
        for column in NUMERIC_COLS
        if column in df.columns
    ]

    if numeric_cols:

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        corr = df[numeric_cols].corr()

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            ax=ax,
            annot=False
        )

        ax.set_title(
            "Feature Correlation Heatmap"
        )

        fname = (
            "correlation_heatmap.png"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                CHARTS_DIR,
                fname
            ),
            dpi=110
        )

        plt.close(fig)

        filenames.append(fname)


    # ========================================================
    # 4. PLACEMENT RATE BY COLLEGE TIER
    # ========================================================

    if (
        "CollegeTier" in df.columns
        and TARGET_COL in df.columns
    ):

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        rate = (
            df.groupby(
                "CollegeTier"
            )[TARGET_COL]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        sns.barplot(
            x=rate.index,
            y=rate.values,
            hue=rate.index,
            ax=ax,
            palette="viridis",
            legend=False
        )

        ax.set_title(
            "Placement Rate by College Tier"
        )

        ax.set_xlabel(
            "College Tier"
        )

        ax.set_ylabel(
            "Placement Rate"
        )

        fname = (
            "placement_by_college_tier.png"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                CHARTS_DIR,
                fname
            ),
            dpi=110
        )

        plt.close(fig)

        filenames.append(fname)


    # ========================================================
    # 5. PLACEMENT RATE BY INTERNSHIPS
    # ========================================================

    if (
        "Internships" in df.columns
        and TARGET_COL in df.columns
    ):

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        rate = (
            df.groupby(
                "Internships"
            )[TARGET_COL]
            .mean()
        )

        sns.barplot(
            x=rate.index,
            y=rate.values,
            hue=rate.index,
            ax=ax,
            palette="viridis",
            legend=False
        )

        ax.set_title(
            "Placement Rate by Number of Internships"
        )

        ax.set_xlabel(
            "Internships"
        )

        ax.set_ylabel(
            "Placement Rate"
        )

        fname = (
            "placement_by_internships.png"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                CHARTS_DIR,
                fname
            ),
            dpi=110
        )

        plt.close(fig)

        filenames.append(fname)


    return filenames


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    from ml.data_loader import (
        load_data,
        clean_data
    )

    # Load dataset
    raw = load_data()

    # Clean dataset
    cleaned = clean_data(raw)

    # Get summary
    stats = get_summary_stats(cleaned)

    print(
        "\nDataset Shape:"
    )

    print(
        stats["shape"]
    )

    print(
        "\nColumns:"
    )

    print(
        stats["columns"]
    )

    print(
        "\nMissing Values:"
    )

    print(
        stats["missing_values"]
    )

    print(
        "\nPlacement Rate:"
    )

    print(
        stats["placement_rate"]
    )

    print(
        "\nCorrelation With Target:"
    )

    print(
        stats["correlation_with_target"]
    )

    # Generate charts
    charts = generate_charts(
        cleaned
    )

    print(
        "\nGenerated Charts:"
    )

    print(
        charts
    )