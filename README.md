# Genomic Variant Classifier: High-Recall SVM Pipeline

License: MIT
Python Version: 3.12+

This project provides a high-performance machine learning pipeline for classifying genomic variants as Pathogenic or Benign using ClinVar data. It is specifically designed to overcome the computational bottlenecks associated with large-scale genomic datasets and Support Vector Machines (SVMs).

## The Problem
Standard Support Vector Machines with RBF kernels scale at O(n^3), making them impossible to train on standard hardware when dealing with hundreds of thousands of genomic variants. Additionally, genomic data is often highly imbalanced (Benign variants vastly outnumber Pathogenic ones), causing traditional models to suffer from zero-recall issues for the pathogenic class.

## The Solution
This pipeline implements an optimized "Safety-First" architecture:

1. Nystroem Kernel Approximation: Maps data into a lower-dimensional space to approximate an RBF kernel, shifting complexity from O(n^3) to O(n) linear scaling.
2. Stochastic Gradient Descent (SGD) Solver: Utilizes an SGD-based SVM implementation to maintain a minimal memory footprint during training.
3. Intel Extension for Scikit-learn: Integrates hardware acceleration to maximize training speed on Intel CPUs.
4. Recall-Specific Optimization: Uses F2-scoring and asymmetric class weighting to prioritize the detection of pathogenic variants, ensuring a high level of clinical sensitivity.

## Performance Metrics
The model was evaluated on a balanced test set derived from ClinVar features.

| Metric | Score |
| :--- | :--- |
| Pathogenic Recall (Sensitivity) | 0.95 |
| Pathogenic Precision | 0.80 |
| Overall Accuracy | 0.86 |
| F1-Score (Macro Average) | 0.86 |

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/svm_project.git
   cd svm_project

2. Set up a virtual environment:
   python -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

## Usage

1. Feature Extraction
Extracts allele frequencies from multiple sources (ESP, ExAC, TGP) and maps molecular consequence severity.
python src/extract_features.py

2. Model Training
Runs a randomized hyperparameter search optimized for F2-score and saves the final model.
python src/train_model.py

## Project Structure
- src/extract_features.py: VCF parsing and biological feature engineering.
- src/train_model.py: Memory-efficient SVM training and hyperparameter tuning.
- data/processed/: Storage for extracted feature CSVs.
- models/: Storage for trained .pkl pipeline files.

## Technical Challenges Overcome

1. Memory Management: Resolved std::bad_alloc (MemoryError) by replacing exact SVC with a Nystroem-approximated SGD classifier, allowing training on 350k+ variants with 32GB of RAM.
2. Class Imbalance: Solved the 0% recall problem through balanced down-sampling and a 3:1 pathogenic-to-benign class weighting strategy.
3. WSL2 Optimization: Migrated project files to the native Linux filesystem (~/) to bypass the slow translation layer of the Windows mount (/mnt/c/).

## License
Distributed under the MIT License. See LICENSE for more information.
