# 📄 Case Study: Twitter Spam Detection using LSTM (Encoder-Decoder to Classifier Evolution)

## 🧠 Overview

This case study explores the design and evolution of a Twitter Spam Detection system, built using deep learning techniques. The system classifies tweets as **Spam** or **Ham (legit)**, using textual features only.

Originally conceived as an encoder-decoder architecture, the project evolved into a more effective LSTM-based binary classifier. This transition significantly improved accuracy, simplified deployment, and helped me understand how architecture selection directly impacts model performance.

---

## 🔍 Problem Statement

With the rise in bot accounts, scam messages, and promotional spam on Twitter, detecting such messages automatically is essential. The goal is to:
> **Classify a tweet into "Spam" or "Ham" using its raw text**

This task involves:
- Understanding natural language
- Handling short and noisy text (tweets)
- Building a model that generalizes well to unseen examples

---

## 🧪 Phase 1: Encoder-Decoder Model

### 🏗️ Architecture
- **Encoder**: LSTM processed tweet input → vector
- **Decoder**: Dense layer attempted to classify from vector

### 🔬 Observations
- **Accuracy**: ~83–91% (Validation)
- **Confidence**: Many outputs hovered around 0.49–0.51
- **Confusion Matrix**: Misclassified obvious spam as ham
- **Training/Deployment**: Slow training, hard to debug in layers
- **Conclusion**: Overkill for binary classification

---

## ✅ Phase 2: LSTM Binary Classifier (Final Version)

### 🏗️ Architecture
Input → Embedding → LSTM → Dropout → LSTM → Dropout → Dense (sigmoid)


- Input: Preprocessed and padded tweet sequences
- Output: Binary probability
- Optimized for binary crossentropy loss

### 🔬 Results
- **Validation Accuracy**: 98.3%
- **False Positives**: 0
- **False Negatives**: 19
- **Model Confidence**: High and stable
- **Training Time**: Faster
- **Deployment**: Integrated easily with Streamlit

---

## 🔁 Model Comparison

| Metric                     | Encoder-Decoder | LSTM Classifier |
|----------------------------|-----------------|------------------|
| Complexity                 | High             | ✅ Simple         |
| Accuracy (Validation)      | ~83–91%          | ✅ 98.3%          |
| Training Stability         | ❌ Unstable       | ✅ Stable         |
| Confidence Distribution    | ❌ Narrow range   | ✅ Clear signal   |
| Deployment Feasibility     | ❌ Difficult      | ✅ Easy w/ Streamlit |
| Overfitting Risk           | High             | ✅ Mitigated w/ Dropout |

---

## 📊 Evaluation Metrics

- Accuracy / Loss curves
- Confusion matrix:
[[965 0] [ 19 106]]


- Classification report:
- F1 Score (Spam): 0.92
- Precision: 1.00
- Recall: 0.85

---

## 🌐 Deployment: Streamlit App

To make the project interactive, a **Streamlit web app** was built:
- Enter any tweet
- See real-time prediction and confidence score
- Optionally visualize model performance (confusion matrix, classification report)

---

## 📚 Key Learnings

- Architecture should match task complexity  
- Evaluation > accuracy — metrics like recall, F1, confusion matrix matter  
- Simpler models often generalize better in production  
- Deployment (Streamlit) teaches valuable lessons beyond model building  

---

## 🔮 Future Enhancements

- Integrate **Twitter API** for live tweet analysis
- Train with **BERT or transformer-based encoders**
- Add **SHAP/LIME explainability** to visualize attention
- Expand dataset with multilingual tweets
- Deploy to **Streamlit Cloud** or **Hugging Face Spaces**

---

## 👋 Final Thoughts

This project taught me the importance of **iterating based on results**, not assumptions. By moving from an encoder-decoder architecture to a streamlined LSTM classifier, I built a spam detection system that performs accurately and is ready for real-world use — all wrapped in a beautiful web interface.

> Every experiment taught me something — and I now understand not just how to build a model, but when to change one.

---

✅ [Back to Main Project README](README.md)




