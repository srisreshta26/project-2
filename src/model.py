
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train_models(df):
    """Train and evaluate two spam classification models."""

    X = df["message"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_types = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    }

    trained_models = {}
    results = []

    for name, classifier in model_types.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True,
            )),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        trained_models[name] = pipeline

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test, predictions, pos_label="spam",
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test, predictions, pos_label="spam",
                zero_division=0,
            ),
            "F1 Score": f1_score(
                y_test, predictions, pos_label="spam",
                zero_division=0,
            ),
            "Confusion Matrix": confusion_matrix(
                y_test, predictions, labels=["ham", "spam"]
            ),
            "Classification Report": classification_report(
                y_test, predictions, zero_division=0
            ),
        })

    return trained_models, results, X_test, y_test
