from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("data/raw/iris.csv")
OUTPUT_DIR = Path("outputs")
METRICS_PATH = OUTPUT_DIR / "metrics.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the Iris dataset from a local CSV file.

    Args:
        data_path: Path to the local Iris CSV file.

    Returns:
        A pandas DataFrame containing the Iris dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run src/download_data.py first."
        )

    return pd.read_csv(data_path)

def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into input features and target labels.

    Args:
        dataset: Iris dataset with feature columns and species column.

    Returns:
        A tuple containing features X and target y.
    """

    feature_columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]

    X = dataset[feature_columns]
    y = dataset["species"]

    return X, y

def train_and_evaluate() -> None:
    """Train a Decision Tree model on Iris and save evaluation outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATA_PATH)
    X, y = split_features_target(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    METRICS_PATH.write_text(f"accuracy={accuracy:.4f}\n", encoding="utf-8")

    display = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    display.figure_.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close(display.figure_)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    
if __name__ == "__main__":
    train_and_evaluate()
    
