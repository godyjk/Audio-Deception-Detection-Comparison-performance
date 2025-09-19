# Audio-Deception-Detection-Comparison-performance
各常见ai模型对音频欺骗检测效果对比以及原因论证
# Audio Deception Detection Project

## Project Overview
This project focuses on detecting deceptive audio content by extracting and analyzing various audio features. Using machine learning techniques, specifically a decision tree classifier optimized via grid search, we aim to distinguish between true and false audio stories.

## Data
- **Source**: The training materials are derived from Queen Mary University of London (QM) public audio files, ensuring the dataset's academic authenticity and reliability.
- **Metadata File**: `Deception/CBU0521DD_stories_attributes.csv` contains information about audio files, including a `Story_type` column (labeled as "True Story" or "False Story") and a `Language` column.
- **Audio Files**: Located in `Deception/CBU0521DD_stories`, there are 100 WAV files that are processed to extract features.

## Feature Extraction
The `extract_features_v2` function extracts a comprehensive set of audio features:
- **Time-domain features**: Zero-crossing rate, energy.
- **MFCC (Mel-frequency cepstral coefficients)**: 20 MFCC coefficients.
- **Spectral features**: Spectral centroid, spectral bandwidth, spectral rolloff.
- **Chroma features**: 12 chroma values.
- **Mel spectrogram features**: Mean values from the mel spectrogram.

## Model Training and Evaluation
1. **Data Preprocessing**:
   - The `Story_type` column is binarized (1 for "True Story", 0 for "False Story").
   - The `Language` column is encoded as categorical codes.
   - The dataset is split into training (70%) and testing (30%) sets.
2. **Model Optimization**: A grid search is performed on a decision tree classifier to find the best hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`).
3. **Evaluation**: The best model from grid search is evaluated using accuracy and a classification report (precision, recall, F1-score) on the test set.

## Usage
1. Ensure required libraries are installed: `pandas`, `numpy`, `scikit-learn`, `librosa`.
2. Place the metadata CSV and audio folder in the correct paths as specified in the code.
3. Run the Jupyter notebook cells sequentially to:
   - Extract features from audio files.
   - Train and optimize the decision tree model.
   - Evaluate the model's performance.

## Results
The grid search finds the best hyperparameters for the decision tree, and the model is evaluated on the test set. Key results include cross-validation accuracy, test accuracy, and a detailed classification report showing performance for both "True Story" and "False Story" classes.

## Dependencies
- Python 3.6+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `librosa`
