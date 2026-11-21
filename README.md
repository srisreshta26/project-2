# Signal – SMS Spam Detection

An end-to-end Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP).

## 🚀 Live Demo

[Try Signal – SMS Spam Detector](https://yh9enai8efsk.streamlit.app/)

## 📌 Project Overview

Signal is a machine learning application designed to identify unwanted or fraudulent SMS messages.

Users can enter a message and receive a prediction indicating whether it is spam or legitimate.

## ✨ Features

- SMS spam classification
- Interactive Streamlit web application
- Text preprocessing using NLP techniques
- Machine Learning model predictions
- Easy-to-use interface

## 🧠 Machine Learning

The project uses:
- Naive Bayes
- A second classification model (see implementation in `src/`)

## 📊 Dataset

The project uses an SMS Spam Collection dataset containing 5,572 messages.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- Streamlit

## 📁 Project Structure

```text
signal-sms-spam-detector/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
├── scripts/
└── tests/
```

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/srisreshta26/signal-sms-spam-detector.git
```

Navigate into the project:

```bash
cd signal-sms-spam-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🎯 Project Goal

To demonstrate how Machine Learning and NLP can be used to identify spam messages and build an accessible, interactive prediction application.

---

**Developed as an AI/ML portfolio project.**