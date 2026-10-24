
import pandas as pd

from src.model import train_models


def test_models_train_and_predict():
    df = pd.DataFrame({
        "message": [
            "Hey, are we meeting today?",
            "Don't forget your homework.",
            "Are you coming to lunch?",
            "I'll call you later.",
            "WIN a free prize now",
            "Claim your cash reward today",
            "Congratulations, you won a prize",
            "URGENT! Claim your free gift",
            "Can you send me the notes?",
            "See you at college tomorrow?",
            "Click now to claim your reward",
            "Free entry in our prize draw",
            "Please bring my notebook.",
            "Your appointment is tomorrow.",
            "You have won free tickets!",
            "Limited offer, claim cash now",
            "Let's study together tonight.",
            "Call me when you reach home.",
            "Exclusive offer, win money now",
            "Free prize waiting for you",
        ],
        "label": [
            "ham", "ham", "ham", "ham",
            "spam", "spam", "spam", "spam",
            "ham", "ham", "spam", "spam",
            "ham", "ham", "spam", "spam",
            "ham", "ham", "spam", "spam",
        ],
    })

    models, results, X_test, y_test = train_models(df)

    assert len(models) == 2
    assert len(results) == 2

    for model in models.values():
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({"ham", "spam"})
