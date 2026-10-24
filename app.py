
import streamlit as st
import pandas as pd

from src.data import load_dataset
from src.model import train_models


st.set_page_config(
    page_title="Signal | Spam Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Signal")
st.subheader("Intelligent SMS Spam Detection")
st.write(
    "Check suspicious messages with machine learning. "
    "Signal compares two models to help identify unwanted messages."
)

@st.cache_data
def get_data():
    return load_dataset()


@st.cache_resource
def get_models():
    data = get_data()
    models, results, X_test, y_test = train_models(data)
    return models, results, X_test, y_test


try:
    df = get_data()
    models, results, X_test, y_test = get_models()
except Exception as error:
    st.error(f"Could not load the dataset or train the models: {error}")
    st.stop()


st.success(f"Dataset ready — {len(df):,} messages loaded.")

model_names = list(models.keys())
selected_model = st.selectbox(
    "Choose your model",
    model_names,
)

st.divider()

st.markdown("### Scan a message")
message = st.text_area(
    "Paste an SMS message below",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150,
)

if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = models[selected_model].predict([message])[0]

        if prediction == "spam":
            st.error("⚠️ Signal classifies this message as SPAM.")
            st.write(
                "Be careful with unexpected prizes, urgent requests, "
                "suspicious links, or requests for personal information."
            )
        else:
            st.success("✅ Signal classifies this message as NORMAL.")
            st.write(
                "This model did not identify the message as spam. "
                "That does not guarantee the message is safe."
            )

st.divider()

st.markdown("### Model performance")
st.caption(
    "Scores below come from a held-out test split of the public dataset."
)

metrics_df = pd.DataFrame([
    {
        key: value
        for key, value in result.items()
        if key not in ("Confusion Matrix", "Classification Report")
    }
    for result in results
])

st.dataframe(
    metrics_df.style.format({
        "Accuracy": "{:.1%}",
        "Balanced Accuracy": "{:.1%}",
        "Precision": "{:.1%}",
        "Recall": "{:.1%}",
        "F1 Score": "{:.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)

chosen_result = next(
    result for result in results
    if result["Model"] == selected_model
)

st.markdown(f"#### {selected_model} — confusion matrix")

matrix = chosen_result["Confusion Matrix"]

st.dataframe(
    pd.DataFrame(
        matrix,
        index=["Actual normal", "Actual spam"],
        columns=["Predicted normal", "Predicted spam"],
    ),
    use_container_width=True,
)

with st.expander("View classification report"):
    st.code(chosen_result["Classification Report"])

st.divider()
st.caption(
    "Signal is an educational machine-learning project. "
    "Predictions can be wrong; verify suspicious messages independently."
)

st.markdown("### Model comparison")

chart_df = metrics_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

st.bar_chart(chart_df)
