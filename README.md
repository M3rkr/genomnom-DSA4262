# Genomnom-DSA4262

This repository is designed for genomic data processing, model training, and evaluation as part of the DSA4262 project. It offers scripts for data preparation, feature selection, model training, predictions, and result visualization.

## Table of Contents

- [Genomnom-DSA4262](#genomnom-dsa4262)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
    - [Key Files and Folders](#key-files-and-folders)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training](#model-training)
    - [Inference](#inference)
    - [Evaluation](#evaluation)
    - [Visualization](#visualization)
    - [Full Pipeline](#full-pipeline)
  - [Pipeline Overview](#pipeline-overview)
  - [Scripts and Utilities](#scripts-and-utilities)
    - [Data Processing and Model Scripts](#data-processing-and-model-scripts)
    - [Utilities](#utilities)
  - [Data Files](#data-files)
  - [Model Files](#model-files)
  - [Results and Visualizations](#results-and-visualizations)

## Project Structure

```plaintext
genomnom-DSA4262/
├── README.md
├── main.py
├── requirements.txt
├── data/
│   ├── training/
│   │   ├── dataset0.json          # **Need to add manually**
│   │   └── data.info.labelled
│   └── prediction/
│       ├── dataset0.json          # **Need to add manually**
│       ├── dataset1.json          # **Need to add manually**
│       ├── dataset2.json          # **Need to add manually**
│       ├── dataset0_prediction.csv
│       ├── dataset1_prediction.csv
│       └── dataset2_prediction.csv
├── graphs/
│   ├── feature_importances.png
│   ├── pr_curve.png
│   ├── roc_curve.png
│   └── training_progress.png
├── model/
│   ├── aggregated_data_selected.csv
│   ├── best_aggregated_model.pth
│   ├── best_hyperparams.json
│   ├── feature_selector.pkl
│   ├── label_encoder_5mer.pkl
│   ├── minmax_scaler.pkl
│   └── training_log.json
├── scripts/
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_selection.py
│   ├── inference.py
│   ├── model_training.py
│   ├── visualization.py
│   └── misc.ipynb
└── utils/
    ├── dataset.py
    └── utils.py
```

### Key Files and Folders

- **data/**: Holds training data, labeled data, and prediction outputs.
  - **Important**: Place `dataset0.json` in both `data/training/` and `data/prediction/` directories, `dataset1.json` and `dataset2.json` in the `data/prediction/` directory before starting the pipeline.
- **graphs/**: Stores visual results, such as PR and ROC curves, and feature importances.
- **model/**: Contains trained model files, hyperparameters, and logs.
- **scripts/**: Includes individual Python scripts for different stages of the pipeline.
- **utils/**: Helper scripts for dataset handling and other utilities.
- **main.py**: Main entry point for running the entire pipeline.

## Setup and Installation

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/M3rkr/genomnom-DSA4262.git
   cd genomnom-DSA4262
   ```
2. **Install dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Before starting the pipeline, ensure that `dataset0.json` is placed in both the `data/training/` and `data/prediction/` directories, and `dataset1.json` and `dataset2.json` are placed in the `data/prediction/` directory.

- **Place `dataset0.json` in `data/training/`**:
  
  ```plaintext
  data/training/dataset0.json
  ```
- **Place `dataset0.json`, `dataset1.json`, and `dataset2.json` in `data/prediction/`**:
  
  ```plaintext
  data/prediction/dataset0.json
  data/prediction/dataset1.json
  data/prediction/dataset2.json
  ```

### Data Preprocessing

To preprocess the data, use:

```bash
python scripts/data_preprocessing.py
```

**Inputs**:

- `data/training/dataset0.json`: Raw training data in JSON format.
- `data/training/data.info.labelled`: Label information for training data.

**Outputs**:

- `model/aggregated_data.csv`: Aggregated and encoded data ready for feature selection and training.

### Model Training

To train the model and optimize hyperparameters:

```bash
python scripts/model_training.py
```

**Inputs**:

- `model/aggregated_data_selected.csv`: Data selected for training after feature selection.

**Outputs**:

- `model/best_aggregated_model.pth`: Trained model with the best parameters.
- `model/best_hyperparams.json`: Hyperparameters of the best model.
- `model/training_log.json`: Training progress log.

### Inference

To run inference on new data:

```bash
python scripts/inference.py --input_file data/prediction/datasetX.json --output_file data/prediction/datasetX_prediction.csv
```

**Inputs**:

- `data/prediction/datasetX.json`: New data for predictions.

**Outputs**:

- `data/prediction/datasetX_prediction.csv`: Prediction scores.

### Evaluation

To evaluate model performance:

```bash
python scripts/evaluation.py
```

**Outputs**:

- `graphs/roc_curve.png`: ROC curve.
- `graphs/pr_curve.png`: Precision-Recall curve.
- Console output of performance metrics.

### Visualization

To plot training progress:

```bash
python scripts/visualization.py
```

**Outputs**:

- `graphs/training_progress.png`: Graph of training progress for ROC-AUC and PR-AUC over epochs.

### Full Pipeline

To run the entire pipeline, execute:

```bash
python main.py
```

## Pipeline Overview

The pipeline includes:

1. **Data Preprocessing**: Cleans and encodes the dataset for feature selection.
2. **Feature Selection**: Selects key features based on model relevance.
3. **Model Training**: Trains the model using optimal hyperparameters.
4. **Evaluation**: Assesses model performance and generates metrics.
5. **Inference**: Generates predictions on new datasets.
6. **Visualization**: Plots the training progress and other results.

## Scripts and Utilities

### Data Processing and Model Scripts

Each script is located in the `scripts/` folder and performs a unique function:

- `data_preprocessing.py`: Prepares raw data for feature selection and training.
- `model_training.py`: Trains and tunes the model.
- `inference.py`: Handles new data predictions.
- `evaluation.py`: Provides performance metrics.
- `visualization.py`: Generates graphs for training progress.

### Utilities

Located in `utils/`, these support data handling and model functions.

## Data Files

- `data/training/dataset0.json`: Raw training data.
- `data/training/data.info.labelled`: Label file for training data.
- `data/prediction/dataset0.json`: Raw prediction data.
- `data/prediction/`: Holds predictions generated by the model.

## Model Files

- `model/best_aggregated_model.pth`: Best-trained model.
- `model/best_hyperparams.json`: Best model hyperparameters.
- `model/training_log.json`: Training log for plotting.

## Results and Visualizations

The `graphs/` folder contains:

- **feature_importances.png**: Feature importance scores.
- **roc_curve.png** and **pr_curve.png**: Performance graphs.
- **training_progress.png**: Graph of ROC-AUC and PR-AUC over epochs.

---

**Note**: Ensure that `dataset0.json` is correctly placed in both `data/training/` and `data/prediction/` directories, and `dataset1.json` and `dataset2.json` are placed in the `data/prediction/` directory before initiating the pipeline to avoid any runtime issues.

