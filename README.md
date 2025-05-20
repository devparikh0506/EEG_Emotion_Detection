# Emotion Detection using DEAP dataset

## Project Overview

This project implements emotion detection using EEG data from the DEAP dataset. The goal is to classify EEG signals into different emotional states, specifically valence and arousal. The project utilizes a deep recurrent neural network architecture called ChronoNet.

## Dataset

The DEAP (Database for Emotion Analysis using Physiological signals) dataset is a multimodal dataset used for analyzing human affective states. It contains EEG and peripheral physiological signals recorded from 32 participants as they watched music videos. The participants rated each video in terms of valence, arousal, dominance, and liking.

*   **Source:** [https://www.eecs.qmul.ac.uk/mmv/datasets/deap/](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
*   **Data:** EEG signals (32 channels, 128 Hz sampling rate)
*   **Labels:** Valence and Arousal (continuous values from 1 to 9)

### Preprocessing Steps

1.  **Data Loading:** EEG data is loaded from `.npy` files.
2.  **Normalization:** EEG signals are z-score normalized across all trials.
3.  **Label Conversion:** Valence and arousal labels are converted into binary classes (Low/High) based on a threshold of 5.
4.  **Data Augmentation:** Data augmentation techniques are applied to increase the size and diversity of the training data.

## Model Architecture

The project uses a deep recurrent neural network architecture called ChronoNet. The ChronoNet model consists of the following layers:

1.  **InceptionBlock1D:** Inception-style convolutional layers for feature extraction.
2.  **GRU Layers:** Gated recurrent unit layers for capturing temporal dependencies.
3.  **Fully Connected Layers:** Fully connected layers for classification.

## Training Process

The model is trained using the following parameters:

*   **Loss Function:** Binary Cross Entropy with Logits Loss
*   **Optimizer:** AdamW
*   **Learning Rate:** 1e-3
*   **Weight Decay:** 1e-4
*   **Learning Rate Scheduler:** ReduceLROnPlateau
*   **Early Stopping:** Patience = 20

## Results

The project achieves the following results:

*   Classification Report for Valence:

```
              precision    recall  f1-score   support

Low Valence       0.60      0.68      0.64       400
High Valence       0.65      0.57      0.61       400

accuracy                           0.62       800
macro avg       0.62      0.62      0.62       800
weighted avg       0.62      0.62      0.62       800
```

*   Classification Report for Arousal:

```
              precision    recall  f1-score   support

Low Arousal       0.62      0.65      0.63       400
High Arousal       0.63      0.60      0.61       400

accuracy                           0.62       800
macro avg       0.62      0.62      0.62       800
weighted avg       0.62      0.62      0.62       800
```

*   Confusion Matrix for Valence:

```
[[272  94]
 [128 306]]
```

*   Confusion Matrix for Arousal:

```
[[259 107]
 [141 293]]
```

## Dependencies

The project requires the following dependencies:

*   pandas
*   numpy
*   matplotlib
*   scikit-learn
*   tqdm
*   scipy
*   torch

## Usage

To run the project, follow these steps:

1.  Download the DEAP dataset from [https://www.eecs.qmul.ac.uk/mmv/datasets/deap/](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and extract the `data_preprocessed_python.zip` file into the `data` directory.
2.  Install the dependencies using `pip install -r requirements.txt`.
3.  Run the `main.ipynb` or `train.ipynb` notebook to train the model and evaluate its performance.

## File Structure

```
├── checkpoints/
├── configs/
│   └── runtime.py
├── data/
│   ├── data_preprocessed_python.zip
│   └── DEAP/
├── datasets/
│   └── deap.py
├── models/
│   └── chrononet.py
├── results/
├── scripts/
│   ├── metrics.py
│   ├── preprocess.py
│   ├── test.py
│   └── train.py
├── utils/
│   ├── data_augmentation.py
│   ├── early_stopping.py
│   └── torch_utils.py
├── main.ipynb
├── README.md
└── requirements.txt
