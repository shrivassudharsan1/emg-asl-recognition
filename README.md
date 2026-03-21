# EMG-Based ASL Recognition

Built a machine learning system to classify ASL gestures from EMG (electromyography) signals using feature engineering and supervised learning.

## Overview
This project focuses on translating muscle activation signals into gesture classifications, enabling real-time human-computer interaction.

## Approach
- Extracted time-domain and frequency-domain EMG features (RMS, MAV, spectral features)
- Trained a Random Forest classifier for gesture recognition
- Evaluated performance using accuracy and F1-score
- Designed pipeline for generalization across users and signal variability

## Results
- Achieved ~95% test accuracy across gesture classes  
- Demonstrated strong performance on structured EMG signals  
- Ongoing work: optimizing for real-time inference and edge deployment  

## Pipeline
EMG Signals → Feature Extraction → Model → Gesture Prediction

## Note
This repository is a high-level project showcase. Full implementation and datasets are part of a collaborative research project and are not included.
