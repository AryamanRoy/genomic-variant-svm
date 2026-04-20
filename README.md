# Genomic Variant Classifier: High-Recall SVM Pipeline

License: MIT  
Python Version: 3.12+

This project provides a high-performance machine learning pipeline for classifying genomic variants as Pathogenic or Benign using ClinVar data. It is specifically designed to overcome the computational bottlenecks associated with large-scale genomic datasets and Support Vector Machines (SVMs).

## Data Acquisition

The model is trained on data provided by ClinVar, a public archive of reports of the relationships among human variations and phenotypes maintained by the National Center for Biotechnology Information (NCBI).

To run this pipeline, you must obtain the ClinVar VCF file:
1. Download the latest clinvar.vcf.gz from the NCBI FTP server: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/
2. Place the downloaded file into the data/raw/ directory.
3. The extraction script expects the file to be named clinvar.vcf.gz.

## Project Structure

The repository is organized to separate source code from raw data and generated assets. Note that the data/ and models/ directories are included in the .gitignore to prevent large file uploads.

```text
svm_project/
├── data/
│   ├── raw/                # Store clinvar.vcf.gz here
│   └── processed/          # Generated feature CSV files
├── models/                 # Storage for trained .pkl pipeline files
├── src/
│   ├── extract_features.py # VCF parsing and feature engineering
│   ├── train_model.py      # Standard SVM training and tuning
│   └── train_model_recall.py # Recall-optimized training (F2-score)
├── .gitignore              # Rules for excluding data/venv from Git
├── LICENSE                 # MIT License details
├── README.md               # Project documentation
└── requirements.txt        # Python library dependencies
```

## The Solution

This pipeline implements an optimized "Safety-First" architecture:
1. Nystroem Kernel Approximation: Maps data into a lower-dimensional space to approximate an RBF kernel, shifting complexity from O(n^3) to O(n) linear scaling.
2. Stochastic Gradient Descent (SGD) Solver: Utilizes an SGD-based SVM implementation to maintain a minimal memory footprint during training.
3. Intel Extension for Scikit-learn: Integrates hardware acceleration to maximize training speed on Intel CPUs.
4. Recall-Specific Optimization: Uses F2-scoring and asymmetric class weighting to prioritize the detection of pathogenic variants.

## Performance Metrics

The model was evaluated on a balanced test set derived from ClinVar features.

| Metric | Score |
| :--- | :--- |
| Pathogenic Recall (Sensitivity) | 0.95 |
| Pathogenic Precision | 0.80 |
| Overall Accuracy | 0.86 |
| F1-Score (Macro Average) | 0.86 |

## Installation

Clone the repository:
```bash
git clone [https://github.com/AryamanRoy/genomic-variant-svm.git](https://github.com/AryamanRoy/genomic-variant-svm.git)
cd genomic-variant-svm
```
Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Feature Extraction:
   Extracts allele frequencies from multiple sources (ESP, ExAC, TGP) and maps molecular consequence    severity.
   ```bash
   python src/extract_features.py
   ```
2. Model Training:
   Runs the optimized training script. Use train_model_recall.py for the version optimized for          clinical sensitivity.
   ```bash
   python src/train_model_recall.py
   ```

## Technical Challenges Overcome

1. Memory Management: Resolved std::bad_alloc issues by replacing exact SVC with a Nystroem-approximated SGD classifier, allowing training on 350k+ variants on local hardware.
2. Class Imbalance: Solved the 0% recall problem through balanced down-sampling and a 3:1 pathogenic-to-benign class weighting strategy.
3. WSL2 Optimization: Migrated project files to the native Linux filesystem (~/) to bypass the slow translation layer of the Windows mount (/mnt/c/).

## License

Distributed under the MIT License. See LICENSE for more information.
