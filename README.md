# dnaZyme

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

This work introduces a powerful tool for designing and improving DNAzymes – molecules that can speed up chemical reactions, much like enzymes. Traditionally, finding effective DNAzymes is slow and expensive, relying heavily on lab experiments. This research offers a faster, computer-based alternative to predict how well a DNAzyme will function *before* it’s even made. 

The core functionality lies in accurately predicting the catalytic efficiency of DNAzymes based on their genetic code and surrounding conditions like buffer composition and temperature. It achieves this by learning from a large collection of existing DNAzyme data, identifying key patterns that determine activity. Researchers can input a new DNAzyme sequence and receive a prediction of its performance, guiding design choices for better catalysts.

The ultimate goal is to accelerate the discovery and optimization of DNAzymes for various applications, reducing reliance on time-consuming lab work and enabling more rational design of these valuable biocatalysts.

---

## Repository content

The dnaZyme repository centers around building a computational platform, SequenceCraft, for designing and predicting the activity of DNAzymes. It comprises several key components working in concert.

A central element is a curated database containing over 350 catalytic cores of DNAzymes, gathered from DNAmoreDB. This database provides the foundational data for training and evaluating predictive models.

The project utilizes machine learning models, primarily LightGBM regressors, to predict the catalytic rate constant (kobs) of DNAzymes. These models are trained on features extracted from DNAzyme sequences and their operating conditions. A convolutional autoencoder is also employed for generating latent representations of sequences.

Feature engineering tools are crucial for converting raw sequence data into a format suitable for machine learning. This includes techniques like k-mer analysis, secondary structure descriptors, and the novel autoencoder approach to capture complex sequence characteristics. The code handles standardization of buffer conditions and parameterization of cofactors, ensuring consistent input for the models.

The repository also incorporates utilities for filtering sequences, calculating molecular descriptors using RDKit, and managing data preprocessing steps. These tools streamline the workflow from raw data to model training and prediction.

Finally, a web resource is implied as an output, providing researchers with access to these predictive capabilities for exploratory analysis and DNAzyme design. The overall system aims to accelerate DNAzyme discovery by offering an in silico alternative to traditional experimental methods like SELEX.

---

## Used algorithms

The codebase utilizes several algorithms for predicting DNAzyme catalytic efficiency. 

**LightGBM Regressor:** This is the primary machine learning model used to predict the catalytic rate (kobs) of DNAzymes. It learns from the features extracted from DNAzyme sequences and experimental conditions to establish a relationship between these factors and catalytic activity.

**K-mer Analysis:**  This technique breaks down DNAzyme sequences into smaller, overlapping segments (k-mers) which are then used as features for the model. It helps capture patterns in the sequence that might be related to catalytic function.

**Autoencoder:** A type of neural network used to create a condensed representation (latent embedding) of the DNAzyme sequences. This simplifies the data while preserving important information, providing another set of useful features for prediction.

**Recursive Feature Elimination:**  This method is employed during model training to identify the most relevant features for predicting kobs. It iteratively removes less important features until an optimal subset remains, improving model accuracy and reducing complexity.

**Stratified K-fold Cross-Validation:** This technique assesses how well the model generalizes to new data by splitting the dataset into multiple subsets (folds). The model is trained on some folds and tested on others, repeating this process several times with different combinations of folds. Stratification ensures each fold has a representative distribution of catalytic rates.

**Normalization & Standardization:** These are preprocessing techniques applied to experimental conditions like buffer compositions and cofactor concentrations. They ensure all features are on a similar scale, preventing any single feature from dominating the model.

---
