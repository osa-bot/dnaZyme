# dnaZyme

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

This work introduces a tool to speed up the design of DNAzymes – molecules that can cut RNA. Traditionally, finding and improving these DNAzymes is slow and expensive, relying heavily on lab experiments. Instead, this research offers a way to *predict* how well a DNAzyme will work based solely on its genetic code and reaction conditions. 

The core idea is to use machine learning to learn from existing data about successful DNAzymes. It builds a system that takes the DNA sequence, along with details like temperature and chemical environment, and estimates how quickly it will cut RNA. Sophisticated techniques are used to translate the DNA code into numbers the computer can understand, including methods inspired by image recognition. A key achievement is high prediction accuracy, allowing researchers to explore many potential designs *before* stepping into the lab. The resulting platform, SequenceCraft, is freely available and aims to significantly accelerate research in areas like gene therapy and diagnostics.

---

## Repository content

The dnaZyme repository centers around a machine learning platform called SequenceCraft designed to predict the catalytic activity of DNAzymes, specifically their observed rate constants (kobs). The core components work together as follows:

1. **Database:** A curated database containing over 350 catalytic cores extracted from DNAmoreDB forms the foundation of the project. This provides the training data for the machine learning models.

2. **Feature Engineering Tools:**  A suite of tools transforms DNAzyme sequences into numerical representations suitable for machine learning. These include methods for generating k-mers (short sequence patterns), calculating physicochemical properties using RDKit descriptors, and employing a convolutional autoencoder to create latent representations of the sequences. The `constants.py` file defines key parameters used in feature generation.

3. **Machine Learning Model:** A LightGBM regressor is trained on these features to predict kobs values.  The training process involves data preprocessing (handling imbalance, encoding), model optimization, and validation using cross-validation techniques.

4. **Autoencoder:** This component learns a compressed representation of the DNAzyme sequences, capturing essential structural and chemical information. The autoencoder’s output is combined with other features to improve prediction accuracy.

5. **Utilities & Constants:**  Supporting utilities handle data loading, condition extraction (pH, buffer components), cofactor analysis, and sequence filtering. `constants.py` provides predefined values for similar compounds and feature selection.

The interaction flow begins with the database providing raw DNAzyme sequences. These are then processed by the feature engineering tools to create numerical descriptors. The machine learning model uses these descriptors to predict kobs, while the autoencoder contributes a learned sequence representation that enhances predictive power.  The `run_model.py` script demonstrates how to load a trained model and apply it to new test data.

---

## Used algorithms

The dnaZyme project utilizes several algorithms for predicting DNAzyme catalytic efficiency. 

**LightGBM Regression:** This is the primary machine learning algorithm used to predict reaction rates (kobs). It learns from the features extracted from DNA sequences and experimental conditions to estimate how quickly a DNAzyme will function.

**Autoencoders:** These are neural networks employed to create a condensed, numerical representation of DNA sequences. They capture complex patterns within the sequence data that might not be obvious through simpler methods, essentially learning a 'code' for each sequence.

**K-mer Frequency Analysis:** This technique breaks down DNA sequences into smaller units (k-mers) and counts how often each unit appears. These frequencies become features used by the LightGBM model to understand sequence patterns related to catalytic activity.

**One-Hot Encoding:** Used in the PDF's SequenceCraft platform, this converts nucleotide bases (A, T, C, G) into a numerical format suitable for machine learning algorithms.

**HyenaDNA Embeddings:** Also used within SequenceCraft, these provide another method of converting DNA sequences into numerical representations, capturing long-range dependencies in the sequence.

**GroupShuffleSplit Cross-Validation:** This is a validation technique that ensures the model's performance isn’t biased by similar sequences being grouped together in either training or testing sets. It helps to create more reliable predictions on new data.

The project also uses algorithms for **data preprocessing**, including methods for filtering sequences and extracting relevant experimental parameters like pH and cofactor concentrations, but these are not machine learning algorithms themselves.

---
