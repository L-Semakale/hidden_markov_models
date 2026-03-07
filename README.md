# Human Activity Recognition Using Hidden Markov Models

**Course:** Machine Learning Techniques  
**Group 39:** Terry Manzi & Limpho Elizabeth Semakale  

---

## Project Overview

This project implements a **Hidden Markov Model (HMM)** pipeline to recognize four human activities — **Still, Standing, Walking, and Jumping** — from smartphone motion sensor data. Accelerometer and gyroscope signals were collected using Android smartphones, processed through feature extraction, and modeled using Gaussian Hidden Markov Models.

The goal is to infer the **hidden activity state** from observable motion signals using probabilistic sequence modeling.

The model is trained using the **Baum–Welch algorithm** and decoded using the **Viterbi algorithm**.

**Best model accuracy:** **86.21%**  
**Baseline model accuracy:** **20.69%**

---

## Activities Recognized

The system classifies the following activities:

| Activity | Description |
|--------|-------------|
| Still | Phone placed on a flat stable surface |
| Standing | Phone held steady at waist height |
| Walking | Consistent walking motion |
| Jumping | Repeated vertical jumping |

---

## Data Collection

Motion data was collected using the **Sensor Logger** mobile application.

Each recording lasted **5–10 seconds**, producing time-series measurements from:

- Accelerometer (x, y, z)
- Gyroscope (x, y, z)

### Recording Setup

| Member | Device | App | Sampling Rate |
|------|------|------|------|
| Terry Manzi | Android Phone A | Sensor Logger | ~99 Hz |
| Limpho Elizabeth Semakale | Android Phone B | Sensor Logger | ~100 Hz |

During preprocessing, sampling rates were harmonized to ensure consistency across recordings.

---

## Feature Extraction

Sensor data was segmented using **sliding windows** and several statistical and frequency features were extracted.

### Time-Domain Features

- Mean acceleration
- Variance
- Root Mean Square (RMS)
- Signal Magnitude Area (SMA)

### Frequency-Domain Features

- Dominant frequency
- Spectral energy from Fast Fourier Transform (FFT)

Feature normalization was applied using **Z-score standardization**.

---

## Hidden Markov Model

Each activity is modeled using a **Gaussian Hidden Markov Model**.

### Model Components

| Component | Description |
|----------|-------------|
| Hidden States | Human activities (Still, Standing, Walking, Jumping) |
| Observations | Feature vectors extracted from sensor windows |
| Transition Matrix | Probability of switching between activities |
| Emission Probabilities | Likelihood of observing features given a state |
| Initial State Distribution | Starting probabilities for each activity |

The model parameters are learned using the **Baum–Welch algorithm**, and activity sequences are decoded using the **Viterbi algorithm**.

---

## Repository Structure

```

├── data/
│   ├── jumping/
│   ├── standing/
│   ├── still/
│   └── walking/
│
├── hmm_activity_recognition.ipynb
├── hmm_activity_recognition_optimized.ipynb
└── README.md

```

- `data/` contains labeled sensor recordings for each activity  
- `hmm_activity_recognition.ipynb` contains the baseline implementation  
- `hmm_activity_recognition_optimized.ipynb` contains the improved model  

---

## Installation

Install required dependencies:

```

pip install numpy pandas scipy scikit-learn hmmlearn matplotlib seaborn

```

---

## Running the Notebooks

### Google Colab

1. Upload the `data/` folder to Google Drive at:

```

MyDrive/HMM_ML_Techniques_2/data

```

2. Open the notebook in Colab.
3. Run all cells sequentially.

---

### Running Locally

1. Clone the repository

```

git clone <repository-url>

```

2. Navigate to the project directory

```

cd <repository-folder>

```

3. Update the data path inside the notebook

```

DATA_PATH = "./data"

```

4. Run the notebook cells.

---

## Model Evaluation

The optimized model was evaluated on unseen test recordings.

| Activity | Sensitivity | Specificity | Precision |
|--------|-------------|-------------|-------------|
| Jumping | 100.0% | 89.6% | 66.7% |
| Standing | 100.0% | 92.7% | 85.0% |
| Still | 73.7% | 100.0% | 100.0% |
| Walking | 75.0% | 100.0% | 100.0% |

**Overall Accuracy: 86.21%**

---

## Key Visualizations

The notebook includes:

- Sensor signal plots
- Feature distributions
- Transition probability heatmaps
- Confusion matrix for model evaluation
- Predicted activity sequences using Viterbi decoding

---

## Discussion

The results show that **Jumping and Standing** are easier to detect due to distinct motion patterns, while **Still and Standing** are sometimes harder to distinguish because both involve minimal movement.

Model performance improved significantly after feature engineering and parameter optimization.

---

## Future Improvements

Potential improvements include:

- Collecting larger datasets
- Adding additional sensors (magnetometer, barometer)
- Applying deep learning models such as LSTM networks
- Using hierarchical Hidden Markov Models
