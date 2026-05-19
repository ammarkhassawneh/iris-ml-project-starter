# Iris ML Project Starter

A structured beginner-friendly machine learning project that trains a classification model on the real Iris dataset from the UCI Machine Learning Repository.

This project is intentionally simple from a modeling perspective, but professional in structure. The goal is not only to train a model, but to start moving from notebook-based experimentation toward clean, reproducible, and GitHub-ready AI engineering projects.

---

## Project Overview

The project trains a simple machine learning classifier to predict the species of an Iris flower based on four numerical measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is the Iris species.

This project demonstrates:

- How to organize a small ML project outside a notebook
- How to download a real dataset from a trusted source
- How to separate source code, raw data, and generated outputs
- How to train and evaluate a baseline model
- How to save reproducible evaluation outputs
- How to prepare a clean GitHub repository

---

## Why This Project Matters

Many junior machine learning projects are built entirely inside notebooks. Notebooks are useful for exploration, but real AI engineering work requires structure, reproducibility, and clarity.

This project is the first step toward building production-grade AI systems by introducing a clean project layout:

```text
data/       Local raw data
outputs/    Local generated results
src/        Python source code
```

The project is intentionally small so the focus stays on engineering discipline rather than dataset complexity.

---

## Dataset

The dataset used in this project is the Iris dataset from the UCI Machine Learning Repository.

Source:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

Dataset characteristics:

| Property | Value |
|---|---:|
| Number of samples | 150 |
| Number of input features | 4 |
| Number of target classes | 3 |
| Problem type | Multiclass classification |
| Feature type | Numerical |
| Target column | species |

Target classes:

```text
Iris-setosa
Iris-versicolor
Iris-virginica
```

---

## Project Structure

```text
iris-ml-project-starter/
├── data/
│   └── raw/
│       └── iris.csv              # Downloaded locally, not committed to Git
├── outputs/
│   ├── metrics.txt               # Generated locally, not committed to Git
│   └── confusion_matrix.png      # Generated locally, not committed to Git
├── src/
│   ├── download_data.py          # Downloads the Iris dataset from UCI
│   └── train_model.py            # Trains and evaluates the model
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tools and Libraries

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| venv | Isolated Python environment |
| pandas | Data loading and tabular data handling |
| NumPy | Numerical computing dependency used across ML workflows |
| Matplotlib | Plotting and saving evaluation figures |
| scikit-learn | Model training, splitting data, and evaluation metrics |
| Git | Local version control |
| GitHub | Remote code hosting and portfolio publishing |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ammarkhassawneh/iris-ml-project-starter.git
cd iris-ml-project-starter
```

If you are still working locally and have not pushed the repository yet, simply open the project folder:

```bash
cd ~/projects/iris-ml-project-starter
```

---

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Verify that the correct Python environment is active:

```bash
which python
```

The output should point to:

```text
.venv/bin/python
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import pandas, numpy, matplotlib, sklearn; print('Environment ready')"
```

---

## How to Run the Project

### Step 1: Download the Dataset

```bash
python src/download_data.py
```

Expected output:

```text
Dataset saved to: data/raw/iris.csv
Rows: 150
Columns: ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species']
```

---

### Step 2: Train and Evaluate the Model

```bash
python src/train_model.py
```

Expected output example:

```text
Accuracy: 0.9667
Metrics saved to: outputs/metrics.txt
Confusion matrix saved to: outputs/confusion_matrix.png
```

---

## Baseline Result

The first baseline model uses a `DecisionTreeClassifier` from scikit-learn.

| Metric | Value |
|---|---:|
| Accuracy | 0.9667 |

This result is a baseline, not a final optimized model. The goal of this first version is to establish a clean and reproducible ML project structure.

The result may vary if the model, random seed, train/test split, or preprocessing steps are changed.

---

## Generated Outputs

After running the training script, the following files are generated locally:

```text
outputs/metrics.txt
outputs/confusion_matrix.png
```

These files are not committed to Git by default because they can be regenerated from the source code.

This is intentional: GitHub should contain the code and instructions required to reproduce the results, not every generated artifact.

---

## Security and Data Handling Notes

Even though the Iris dataset is public, this project follows safe data-handling habits:

- Raw data is stored locally under `data/raw/`
- Raw data is excluded from Git using `.gitignore`
- Generated outputs are excluded from Git by default
- No secrets, API keys, or credentials are stored in the repository
- The dataset source is documented clearly for reproducibility

These habits are important because real client projects may involve sensitive, private, regulated, or confidential data.

---

## Current Limitations

This is the first structured version of the project. It intentionally keeps the modeling simple.

Current limitations:

- No automated tests yet
- No configuration file yet
- No model persistence yet
- No command-line interface yet
- No experiment tracking yet
- No data validation layer yet
- No API or deployment layer yet

These features will be introduced gradually in future lessons.

---

## Next Improvements

Planned improvements:

- Refactor scripts into cleaner reusable functions
- Add basic unit tests
- Save the trained model to disk
- Add a configuration file
- Add a simple command-line interface
- Add model evaluation reports
- Add GitHub Actions for automated checks
- Add data validation
- Move gradually toward a production-ready ML project structure

---

## Learning Objective

By completing this project, the learner understands the first step in moving from:

```text
Notebook-based experimentation
```

to:

```text
Structured machine learning engineering
```

This is the foundation for building more advanced AI systems later, including APIs, pipelines, monitoring, Dockerized services, and production-grade ML workflows.

---

## License

This project is released under the MIT License.
