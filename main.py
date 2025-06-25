from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import time
import random
import string
import pandas as pd
import numpy as np

nltk.download('stopwords')

# Reproducibility
SEED = 420 
random.seed(SEED)
np.random.seed(SEED)

# Metrics
def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

SCORING = {
    "F1": f1_score,
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "MCC": matthews_corrcoef,
    "ROC AUC": roc_auc_score,
    "PRC AREA": average_precision_score,
    "FPR": false_positive_rate, 
}

def save_scores(experiment: str, index: str, values: dict) -> None:
    columns = list(SCORING.keys()) + ["training_time", "inference_time"]
    scores = pd.DataFrame(columns=columns)
    row = {}
    for metric in SCORING.keys():
        if metric in values:
            val = values[metric]
            row[metric] = round(val, 4) if isinstance(val, (float, int)) else val
    row["training_time"] = round(values.get("training_time", 0), 4)
    row["inference_time"] = round(values.get("inference_time", 0), 4)
    scores.loc[index] = row
    print(scores)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    filtered = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Apply preprocessing
for df in [train_df, test_df]:
    df['text'] = df['text'].astype(str).apply(preprocess_text)

# Features and labels
X_train, y_train = train_df['text'], train_df['label']
X_test, y_test = test_df['text'], test_df['label']

# Vectorizer
vectorizer = CountVectorizer()

# Classifiers
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate
for name, model in models.items():
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model),
    ])

    start_train = time.time()
    pipeline.fit(X_train, y_train)
    end_train = time.time()

    start_infer = time.time()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    end_infer = time.time()

    metrics = {
        "training_time": end_train - start_train,
        "inference_time": end_infer - start_infer,
    }

    for metric_name, metric_fn in SCORING.items():
        try:
            if metric_name in ["ROC AUC", "PRC AREA"] and y_proba is not None:
                metrics[metric_name] = metric_fn(y_test, y_proba)
            else:
                metrics[metric_name] = metric_fn(y_test, y_pred)
        except Exception as e:
            metrics[metric_name] = None  

    save_scores("Spam Filtering", name, metrics)