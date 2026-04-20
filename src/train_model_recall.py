from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, make_scorer, fbeta_score
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

def train_recall_optimized_svm():
    print("Loading data...")
    df = pd.read_csv("data/processed/variant_features.csv", low_memory=False)
    
    # 1. Balanced Sampling remains the foundation
    pathogenic = df[df['LABEL'] == 1]
    benign = df[df['LABEL'] == 0].sample(n=len(pathogenic), random_state=42)
    df_balanced = pd.concat([benign, pathogenic])
    
    X = df_balanced.drop(columns=['CHROM', 'POS', 'LABEL'])
    y = df_balanced['LABEL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Define F2 Scorer (Prioritizes recall over precision)
    f2_scorer = make_scorer(fbeta_score, beta=2)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('feature_map', Nystroem(random_state=42)),
        ('svm', SGDClassifier(loss='hinge', random_state=42))
    ])

    # 3. Dynamic Class Weighting in the search
    # We let the search find if 3x, 5x, or 10x weighting is best
    param_dist = {
        'feature_map__n_components': [100, 300],
        'svm__alpha': [1e-4, 1e-3, 1e-2],
        'svm__class_weight': [{0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 8}]
    }

    print("Searching for high-recall balance...")
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        n_iter=10, cv=3, scoring=f2_scorer, n_jobs=-1, verbose=1
    )

    search.fit(X_train, y_train)

    print(f"\nBest High-Recall Parameters: {search.best_params_}")

    print("\n--- Model Evaluation ---")
    predictions = search.predict(X_test)
    print(classification_report(y_test, predictions, target_names=['Benign', 'Pathogenic']))

    joblib.dump(search.best_estimator_, "data/processed/svm_recall_optimized.pkl")

if __name__ == "__main__":
    train_recall_optimized_svm()