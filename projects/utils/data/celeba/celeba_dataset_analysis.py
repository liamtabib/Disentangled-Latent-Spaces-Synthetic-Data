import hydra
import pandas as pd
from omegaconf import DictConfig


def general_dataset_info(df, writer):
    df.dtypes.to_excel(writer, sheet_name="Data Types")
    df.duplicated().to_excel(writer, sheet_name="Duplicates")
    df.isnull().sum().to_excel(writer, sheet_name="Missing Values", index=True)
    return writer


def attributes_distribution(df):
    # Create an empty dataframe to store the value_counts results
    value_counts_df = pd.DataFrame()
    # Loop over each column of the dataframe
    for col in df.columns:
        if col not in ["Unnamed: 0", "filename", "id", "split"]:
            value_counts = df[col].value_counts()
            value_counts.name = col
            value_counts_df = value_counts_df.append(value_counts)

    value_counts_df["Percentage of -1"] = (value_counts_df[-1.0] / df.shape[0]) * 100
    value_counts_df["Percentage of 1"] = (value_counts_df[1.0] / df.shape[0]) * 100
    value_counts_df = value_counts_df.reset_index()
    value_counts_df.columns = [
        "Attributes",
        "-1",
        "1",
        "Percentage of -1",
        "Percentage of 1",
    ]
    return value_counts_df


def postprocessing(df):
    # Hair
    df = df[(df["Gray_Hair"] != 1) | (df["Blond_Hair"] != 1)]
    df = df[(df["Gray_Hair"] != 1) | (df["Black_Hair"] != 1)]
    df = df[(df["Gray_Hair"] != 1) | (df["Brown_Hair"] != 1)]

    df = df[(df["Blond_Hair"] != 1) | (df["Black_Hair"] != 1)]
    df = df[(df["Blond_Hair"] != 1) | (df["Brown_Hair"] != 1)]

    df = df[(df["Black_Hair"] != 1) | (df["Brown_Hair"] != 1)]

    df = df[(df["Gray_Hair"] != 1) | (df["Young"] != 1)]

    df = df[(df["Bald"] != 1) | (df["Receding_Hairline"] != 1)]
    df = df[(df["Bald"] != 1) | (df["Bangs"] != 1)]
    df = df[(df["Receding_Hairline"] != 1) | (df["Bangs"] != 1)]
    # Beard
    df = df[(df["Goatee"] != 1) | (df["Male"] != -1)]
    df = df[(df["Goatee"] != 1) | (df["Mustache"] != -1)]
    df = df[(df["Goatee"] != 1) | (df["No_Beard"] != 1)]
    df = df[(df["Mustache"] != 1) | (df["No_Beard"] != 1)]

    df = df[(df["Sideburns"] != 1) | (df["No_Beard"] != 1)]
    # We lose sideburns but we also remove wrong annotation about beard
    df = df.drop("Sideburns", axis=1)
    df["Beard"] = df["No_Beard"] * -1
    df = df.drop("No_Beard", axis=1)
    # Makeup
    df = df[(df["Heavy_Makeup"] != 1) | (df["Male"] != 1)]
    df = df[(df["Heavy_Makeup"] != 1) | (df["Wearing_Lipstick"] != -1)]
    df = df[(df["Heavy_Makeup"] != -1) | (df["Wearing_Lipstick"] != 1)]

    # We can remove wearing lipstick since now these two class are the same
    df = df.drop("Wearing_Lipstick", axis=1)
    df = df.drop("Attractive", axis=1)
    return df


def id_distribution(df, writer):
    unique_split = df["split"].value_counts()
    unique_split.to_excel(writer, sheet_name="Splits")

    unique_id = df["id"].value_counts()
    unique_id.to_excel(writer, sheet_name="Unique IDs")
    ids = [index for index, _ in unique_id.items()]
    id_dist = pd.DataFrame(columns=["id", "train", "val", "test", "Test with ids"])

    for id in ids:
        id_dist = id_dist.append(
            {
                "id": id,
                "train": len(df.loc[(df["id"] == id) & (df["split"] == 0)]),
                "val": len(df.loc[(df["id"] == id) & (df["split"] == 1)]),
                "test": len(df.loc[(df["id"] == id) & (df["split"] == 2)]),
                "Test_with_ids": len(df.loc[(df["id"] == id) & (df["split"] == 3)]),
            },
            ignore_index=True,
        )
    id_dist.to_excel(writer, sheet_name="id_dist")


@hydra.main(config_path="../../../projects/config", config_name="config_FR", version_base="1.1")
def main(cfg: DictConfig) -> None:
    raise RuntimeError("This code may not work anymore with the new structure of the datasets")
    # Load the data and create the Excel report
    data_path = f"{cfg['data_root']}/{cfg['dataset']['dataset_type']}/annotations.csv"
    df = pd.read_csv(data_path)
    writer = pd.ExcelWriter("./../analysis_results_postpro.xlsx", engine="openpyxl")

    df = postprocessing(df)

    writer = general_dataset_info(df, writer)

    id_distribution(df, writer)

    value_counts_df = attributes_distribution(df)
    value_counts_df.to_excel(writer, sheet_name="Attributes count Total")

    value_counts_train = attributes_distribution(df[df["split"] == 0])
    value_counts_train.to_excel(writer, sheet_name="Attributes count Train")

    value_counts_eval = attributes_distribution(df[df["split"] == 1])
    value_counts_eval.to_excel(writer, sheet_name="Attributes count Eval")

    value_counts_test = attributes_distribution(df[df["split"] == 2])
    value_counts_test.to_excel(writer, sheet_name="Attributes count Test")

    value_counts_test = attributes_distribution(df[df["split"] == 3])
    value_counts_test.to_excel(writer, sheet_name="Attributes count Test with ids")

    unique_split = df["split"].value_counts()
    unique_split.to_excel(writer, sheet_name="Splits")

    unique_id = df["id"].value_counts()
    unique_id.to_excel(writer, sheet_name="Unique IDs")

    # Correlation
    df2 = df.drop(columns=["Unnamed: 0", "filename", "id", "split"])

    corr_matrix = df2.corr()
    corr_matrix.to_excel(writer, sheet_name="Corr between attr")

    writer.save()


if __name__ == "__main__":
    main()
