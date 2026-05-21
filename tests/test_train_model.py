from pathlib import Path

import pandas as pd

from src.train_model import save_metrics, split_features_target


def test_split_features_target_returns_expected_shapes() -> None:
    """Test that features and target are split correctly."""

    dataset = pd.DataFrame(
        {
            "sepal_length_cm": [5.1, 4.9],
            "sepal_width_cm": [3.5, 3.0],
            "petal_length_cm": [1.4, 1.4],
            "petal_width_cm": [0.2, 0.2],
            "species": ["Iris-setosa", "Iris-setosa"],
        }
    )

    X, y = split_features_target(dataset)

    assert X.shape == (2, 4)
    assert y.shape == (2,)
    assert list(X.columns) == [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]


def test_save_metrics_writes_accuracy_file(tmp_path: Path) -> None:
    """Test that metrics are written to a text file."""

    metrics_path = tmp_path / "metrics.txt"

    save_metrics(accuracy=0.9667, metrics_path=metrics_path)

    assert metrics_path.exists()
    assert metrics_path.read_text(encoding="utf-8") == "accuracy=0.9667\n"
