# Genomnom-DSA4262

This repository is designed for genomic data processing, model training, and evaluation as part of the DSA4262 project. It offers scripts for data preparation, feature selection, model training, predictions, and result visualization.

## Table of Contents

- Project Structure
- Setup and Installation
- Usage
- Pipeline Overview
- Scripts and Utilities
- Data Files
- Model Files
- Results and Visualizations

## Project Structure

\`\`\`
genomnom-DSA4262/
├── README.md
├── main.py
├── requirements.txt
├── data/
│   ├── training/
│   │   ├── dataset0.json(need to add manually)
│   │   └── data.info.labelled
│   └── prediction/
│       ├── dataset0\_prediction.csv
│       ├── dataset1\_prediction.csv
│       └── dataset2\_prediction.csv
├── graphs/
│   ├── feature\_importances.png
│   ├── pr\_curve.png
│   ├── roc\_curve.png
│   └── training\_progress.png
├── model/
│   ├── aggregated\_data\_selected.csv
│   ├── best\_aggregated\_model.pth
│   ├── best\_hyperparams.json
│   ├── feature\_selector.pkl
│   ├── label\_encoder\_5mer.pkl
│   ├── minmax\_scaler.pkl
│   └── training\_log.json
├── scripts/
│   ├── data\_preprocessing.py
│   ├── evaluation.py
│   ├── feature\_selection.py
│   ├── inference.py
│   ├── model\_training.py
│   ├── visualization.py
│   └── misc.ipynb
└── utils/
├── dataset.py
└── utils.py
\`\`\`

### Key Files and Folders

- \*\*data/\*\*: Holds training data, labeled data, and prediction outputs.
- \*\*graphs/\*\*: Stores visual results, such as PR and ROC curves, and feature importances.
- \*\*model/\*\*: Contains trained model files, hyperparameters, and logs.
- \*\*scripts/\*\*: Includes individual Python scripts for different stages of the pipeline.
- \*\*utils/\*\*: Helper scripts for dataset handling and other utilities.
- \*\*main.py\*\*: Main entry point for running the entire pipeline.

## Setup and Installation

1. \*\*Clone the repository\*\*:

   \`\`\`bash

   git clone https://github.com/M3rkr/genomnom-DSA4262.git

   cd genomnom-DSA4262

   \`\`\`
2. \*\*Install dependencies\*\*:

   \`\`\`bash

   pip install -r requirements.txt

   \`\`\`

## Usage

### Data Preprocessing

To preprocess the data, use:

\`\`\`bash

python scripts/data\_preprocessing.py

\`\`\`

\*\*Inputs\*\*:

- \`data/training/dataset0.json\`: Raw training data in JSON format.
- \`data/training/data.info.labelled\`: Label information for training data.

\*\*Outputs\*\*:

- \`model/aggregated\_data.csv\`: Aggregated and encoded data ready for feature selection and training.

### Model Training

To train the model and optimize hyperparameters:

\`\`\`bash

python scripts/model\_training.py

\`\`\`

\*\*Inputs\*\*:

- \`model/aggregated\_data\_selected.csv\`: Data selected for training after feature selection.

\*\*Outputs\*\*:

- \`model/best\_aggregated\_model.pth\`: Trained model with the best parameters.
- \`model/best\_hyperparams.json\`: Hyperparameters of the best model.
- \`model/training\_log.json\`: Training progress log.

### Inference

To run inference on new data:

\`\`\`bash

python scripts/inference.py --input\_file data/prediction/datasetX.json --output\_file data/prediction/datasetX\_prediction.csv

\`\`\`

\*\*Inputs\*\*:

- \`data/prediction/datasetX.json\`: New data for predictions.

\*\*Outputs\*\*:

- \`data/prediction/datasetX\_prediction.csv\`: Prediction scores.

### Evaluation

To evaluate model performance:

\`\`\`bash

python scripts/evaluation.py

\`\`\`

\*\*Outputs\*\*:

- \`graphs/roc\_curve.png\`: ROC curve.
- \`graphs/pr\_curve.png\`: Precision-Recall curve.
- Console output of performance metrics.

### Visualization

To plot training progress:

\`\`\`bash

python scripts/visualization.py

\`\`\`

\*\*Outputs\*\*:

- \`graphs/training\_progress.png\`: Graph of training progress for ROC-AUC and PR-AUC over epochs.

### Full Pipeline

To run the entire pipeline, execute:

\`\`\`bash

python main.py

\`\`\`

## Pipeline Overview

The pipeline includes:

1. \*\*Data Preprocessing\*\*: Cleans and encodes the dataset for feature selection.
2. \*\*Feature Selection\*\*: Selects key features based on model relevance.
3. \*\*Model Training\*\*: Trains the model using optimal hyperparameters.
4. \*\*Evaluation\*\*: Assesses model performance and generates metrics.
5. \*\*Inference\*\*: Generates predictions on new datasets.
6. \*\*Visualization\*\*: Plots the training progress and other results.

## Scripts and Utilities

### Data Processing and Model Scripts

Each script is located in the \`scripts/\` folder and performs a unique function:

- \`data\_preprocessing.py\`: Prepares raw data for feature selection and training.
- \`model\_training.py\`: Trains and tunes the model.
- \`inference.py\`: Handles new data predictions.
- \`evaluation.py\`: Provides performance metrics.
- \`visualization.py\`: Generates graphs for training progress.

### Utilities

Located in \`utils/\`, these support data handling and model functions.

## Data Files

- \`data/training/dataset0.json\`: Raw training data.
- \`data/training/data.info.labelled\`: Label file for training data.
- \`data/prediction/\`: Holds predictions generated by the model.

## Model Files

- \`model/best\_aggregated\_model.pth\`: Best-trained model.
- \`model/best\_hyperparams.json\`: Best model hyperparameters.
- \`model/training\_log.json\`: Training log for plotting.

## Results and Visualizations

The \`graphs/\` folder contains:

- \*\*feature\_importances.png\*\*: Feature importance scores.
- \*\*roc\_curve.png\*\* and \*\*pr\_curve.png\*\*: Performance graphs.
- \*\*training\_progress.png\*\*: Graph of ROC-AUC and PR-AUC over epochs.

## License

This project is licensed under the MIT License.

---

For any questions, refer to the [AskTheCode Documentation](https://docs.askthecode.ai) for GitHub integration help.
