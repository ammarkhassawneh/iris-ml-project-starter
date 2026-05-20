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

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

TARGET_COLUMN = "species"


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the Iris dataset from a local CSV file.

    Args:
        data_path: Path to the local Iris CSV file.

    Returns:
        A pandas DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run src/download_data.py first."
        )

    return pd.read_csv(data_path)


def validate_columns(dataset: pd.DataFrame) -> None:
    """Validate that the dataset contains the expected columns.

    Args:
        dataset: Input dataset.

    Raises:
        ValueError: If one or more required columns are missing.
    """

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    available_columns = set(dataset.columns)
    missing_columns = required_columns - available_columns

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into input features and target labels.

    Args:
        dataset: Dataset containing feature columns and target column.

    Returns:
        A tuple containing X features and y target labels.
    """

    validate_columns(dataset)

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split.

    Args:
        X: Input features.
        y: Target labels.

    Returns:
        X_train, X_test, y_train, and y_test.
    """

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier.

    Args:
        X_train: Training input features.
        y_train: Training target labels.

    Returns:
        A trained DecisionTreeClassifier.
    """

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.Series, float]:
    """Evaluate the trained model on test data.

    Args:
        model: Trained classifier.
        X_test: Test input features.
        y_test: Test target labels.

    Returns:
        Predictions and accuracy score.
    """

    predictions = pd.Series(model.predict(X_test), index=y_test.index)
    accuracy = accuracy_score(y_test, predictions)

    return predictions, accuracy


def save_metrics(accuracy: float, metrics_path: Path) -> None:
    """Save model metrics to a text file.

    Args:
        accuracy: Model accuracy score.
        metrics_path: Destination file path.
    """

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(f"accuracy={accuracy:.4f}\n", encoding="utf-8")


def save_confusion_matrix(
    y_test: pd.Series,
    predictions: pd.Series,
    output_path: Path,
) -> None:
    """Save a confusion matrix plot as an image file.

    Args:
        y_test: Ground-truth test labels.
        predictions: Model predictions.
        output_path: Destination image path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    display.figure_.savefig(output_path, bbox_inches="tight")
    plt.close(display.figure_)


def run_training_pipeline() -> None:
    """Run the full model training and evaluation workflow."""

    dataset = load_dataset(DATA_PATH)
    X, y = split_features_target(dataset)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    model = train_model(X_train, y_train)
    predictions, accuracy = evaluate_model(model, X_test, y_test)

    save_metrics(accuracy, METRICS_PATH)
    save_confusion_matrix(y_test, predictions, CONFUSION_MATRIX_PATH)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    run_training_pipeline()