from pathlib import Path
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
OUTPUT_PATH = Path("data/raw/iris.csv")

COLUMN_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
    "species",
]


def download_iris_data() -> None:
    """Download the Iris dataset from UCI and save it locally."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {list(dataset.columns)}")


if __name__ == "__main__":
    download_iris_data()
