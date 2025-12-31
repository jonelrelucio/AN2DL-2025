# AN2DL-challenges-2025-26

This repository contains solutions for two AN2DL 2025–26 Kaggle challenges with around 200 participants, covering time-series classification and medical image classification. The projects achieved top-tier Kaggle rankings using deep learning models (RNNs and CNNs), and the accompanying technical reports received excellent evaluations.
- Challenge 1 _(Time Series)_: 14th place — [Kaggle Competition 1](https://www.kaggle.com/competitions/an2dl2526c1)
— [Report 1](challenge1/AN2DL_challenge_1_report.pdf)
- Challenge 2 _(WSI Images)_: 8th place — [Kaggle Competition 2](https://www.kaggle.com/competitions/an2dl2526c2v2)
— [Report 2](challenge2/AN2DL_challenge_2_report.pdf)

For each challenge, combined Kaggle ranking and report evaluation resulted in a full score of 5.5/5.5.

## Challenge 1: Time Series Classification

The challenge involved classifying multivariate time series data to predict pain status (no pain, low pain, high pain) using Recurrent Neural Network architectures. Each sequence consisted of 160 measurements from multiple channels, optimizing for F1-score performance.


### Results

The final model achieved 0.9501 F1-score on the public test set, representing significant improvement over the baseline. While competitive, state-of-the-art models in the competition exceeded 0.9700, indicating room for optimization through enhanced feature engineering, data augmentation, and architectural refinements.
On the private test set, it achieved a **0.96268 F1-Score**, boosting our ranking to **14th** place.


## Challenge 2: Image Classification


The challenge involved classifying low-magnification Whole Slide Imaging (WSI) of human tissue into four molecular subtypes corresponding to potential diseases. The task utilized Convolutional Neural Network architectures, optimizing for F1-score performance.


### Results

The final model achieved an **F1-score of 0.4500**, significantly outperforming the random baseline (0.2932) reaching the **8th** position in the Kaggle competition. The model successfully learns morphological patterns but is severely limited by the Triple Negative class and shows overconfidence in certain predictions.


