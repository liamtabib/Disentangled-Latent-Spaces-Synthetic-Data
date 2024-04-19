import argparse
from pathlib import Path
from typing import List, Union

import pandas as pd
import sweetviz as sv
from pandas_profiling import ProfileReport


def visualize_dataset(
    source: List[Union[pd.DataFrame, str]],
    compare: List[Union[pd.DataFrame, str]],
    skip: List[str],
    target_feat: str,
    json: bool,
) -> None:
    """Visualizes a dataset and creates a html-file of the visualization
    and opens it in the browser.

    Args:
        source (List[pd.DataFrame, str]): List with a Dataframe containing
            the annotations to be visualized. The second element is used to
            name the dataframe.
        compare (List[pd.DataFrame, str]): List with a Dataframe containing
            the annotations to be visualized and compared with `source`.
            The second element is used to name the dataframe.
            If None is given, no comparison will be made.
        skip (List[str]): Contains column names of which to skip.
        target_feat (str): Feature name for which additional association visualization
            will be made. Note that `source` and
            `compare` should have this column. Note that this is only applicable when
            compare is not None.
        json (bool): True if report should be saved as a json file (instead of html).
            Note that this is only applicable when `compare` is None.
    """

    # Preprocess source annotations
    source[0] = _drop_columns(source[0], skip)

    # Analyze/compare dataset(s)
    if compare is None:
        # Visualize using pandas profiling
        profile = ProfileReport(source[0], title=source[1])
        ext = "json" if json else "html"

        # Save report as either html or json
        profile.to_file(f"report_{source[1]}.{ext}")
    else:
        # Preprocess compare annotations
        compare[0] = _drop_columns(compare[0], skip)

        # Visualize using sweetviz
        report = sv.compare(source, compare, target_feat=target_feat)

        # Fetch html
        report.show_html(layout="vertical")


def _drop_columns(df: pd.DataFrame, skip: List[str]):
    """Drop columns specified by `skip`.

    Args:
        df (pd.DataFrame): Dataframe containing the labels.
        skip (List[str]): List of columns that should be skipped.

    Returns:
        pd.DataFrame: Dataframe without the dropped columns.
    """
    for col in skip:
        try:
            df = df.drop(columns=[col])
        except KeyError:
            print(f"Column {col} could not be dropped since it doesn't exist.")
    return df


def _parse_args() -> argparse.Namespace:
    """Parses arguments from user"""
    parser = argparse.ArgumentParser()

    # Source dataset
    parser.add_argument("-d", "--dataset_path", required=True)
    parser.add_argument("-n", "--dataset_name", default="dataset")

    # Compare dataset
    parser.add_argument("-c", "--compare_path", default=None)
    parser.add_argument("-a", "--compare_name", default="compare")

    # Util arguments
    parser.add_argument("-f", "--target_feat", default=None)
    parser.add_argument("--skip", default=["filename", "id", "split"])
    parser.add_argument("-j", "--json", default=0)

    return parser.parse_args()


if __name__ == "__main__":
    print("Fetching data to start the visualization...")

    # Parse arguments
    args = _parse_args()

    # Get the source dataframe
    source_df = pd.read_csv(Path(args.dataset_path))

    # If specified, get the compare dataframe
    compare_df = None if args.compare_path is None else pd.read_csv(Path(args.compare_path))

    print("Preparing visualization html...")
    visualize_dataset(
        [source_df, str(args.dataset_name)],
        compare_df if compare_df is None else [compare_df, str(args.compare_name)],
        args.skip,
        args.target_feat,
        bool(args.json),
    )

    print("Preparations completed!")
    print("Visualization should open automatically in a browser.")
    print("(If not, open the generated html(or json) file manually.)")
