# ITU BDS MLOPS'25 - Project

## Project Overview
This repository contains a production-grade MLOps pipeline for a Lead Scoring model. It demonstrates the transformation of a monolithic notebook into a structured, automated workflow using **Dagger** and **GitHub Actions**.

The solution adheres to the following architecture:
![Project architecture](./docs/project-architecture.png)

## Repository Structure

- **`src/`**: The core Python source code.
    - `data.py`: Data loading and preprocessing.
    - `train.py`: Model training (XGBoost vs Logistic Regression) and selection.
    - `evaluate.py`: Evaluation metrics.
    - `main.py`: The entry point script that orchestrates the pipeline.
- **`ci/`**: The Dagger automation pipeline written in **Go**.
    - `pipeline.go`: Defines the containerized workflow (Mount -> Install -> Train -> Export).
- **`.github/workflows/`**: Continuous Integration configuration.
    - `dagger.yml`: Triggers the Dagger pipeline on GitHub Runners and validates the output.
- **`artifacts/`**: Stores generated outputs (models, scalers) and pulled data.
- **`notebooks/`**: The original exploratory analysis (kept for reference).

## How to Run

### Prerequisities
- **Docker Desktop** (Must be running)
- **Go** (for Dagger)
- **Python 3.11+**

### 1. Run Locally (Python)
You can run the pipeline directly if you have the environment set up:
```bash
pip install -r requirements.txt
python -m src.main
```

### 2. Run Automation (Dagger)
To run the exact same pipeline used in production (containerized):
```bash
cd ci
dagger run go run pipeline.go
```
This will:
1.  Spin up a clean Python container.
2.  Install all dependencies.
3.  Train and select the best model.
4.  Export `model.joblib` to your local `artifacts/` folder.

### 3. CI/CD (GitHub Actions)
Every push to `main` or `feature/*` branches triggers the cloud pipeline:
1.  **Build & Train**: Runs the Dagger pipeline.
2.  **Upload**: Saves the model artifact.
3.  **Validate**: Runs `itu-sdse-project-model-validator` to certify the model.

## Model Details
The pipeline trains two candidate models:
1.  **XGBoost Classifier**
2.  **Logistic Regression**

It automatically selects the best performer based on the **F1 Score** on a held-out test set.

## Members
- [Your Name/Group]
