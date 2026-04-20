from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

def train_tuned_svm():
    print("Loading data...")
    df = pd.read_csv("data/processed/variant_features.csv", low_memory=False)

    pathogenic = df[df['LABEL'] == 1]
    benign = df[df['LABEL'] == 0].sample(n=len(pathogenic), random_state=42)
    df_balanced = pd.concat([benign, pathogenic])
    
    X = df_balanced.drop(columns=['CHROM', 'POS', 'LABEL'])
    y = df_balanced['LABEL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('feature_map', Nystroem(random_state=42)),
        ('svm', SGDClassifier(loss='hinge', class_weight='balanced', random_state=42))
    ])

    param_dist = {
        'feature_map__n_components': [100, 300, 500],
        'feature_map__gamma': [0.01, 0.1, 1.0, None],
        'svm__alpha': [1e-4, 1e-3, 1e-2, 1e-1] # Penalty strength
    }

    print("Starting Hyperparameter Search...")
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        n_iter=10, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1
    )

    search.fit(X_train, y_train)

    print(f"\nBest Parameters Found: {search.best_params_}")

    print("\n--- Model Evaluation ---")
    predictions = search.predict(X_test)
    print(classification_report(y_test, predictions, target_names=['Benign', 'Pathogenic']))

    joblib.dump(search.best_estimator_, "data/processed/svm_variant_pipeline.pkl")
    print("\nOptimized Pipeline saved.")

if __name__ == "__main__":
    train_tuned_svm()