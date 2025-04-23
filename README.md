# 🧠 Twitter Spam Detection using LSTM (Encoder-Decoder to Classifier Evolution)

This project is a deep learning–based spam detection system designed to classify tweets as either Spam or ham (legit). It was initially developed using an Encoder-Decoder architecture, but later optimized into a simpler and more efficient LSTM-based binary classifier. The result is a high-performing, real-time deployable model powered by TensorFlow and Streamlit.

---

## 🧩 Why This Project?

Twitter is a hub of real-time content, but it is also filled with spam, scams, and bot tweets. Building a model that can intelligently identify such content not only improves user safety but also strengthens your NLP and deployment skills.

---

## 🔄 Model Evolution

### 🧪 **Initial Plan: Encoder-Decoder Model**
- Idea: Use an encoder to summarize tweets → decoder to classify
- Why: Inspired by machine translation and text generation tasks

Training & Validation (Encoder)

Metric	Value
Training Accuracy	~97%
Validation Accuracy	~91%
Training Loss	Low (binary_crossentropy)
Validation Loss	Higher than training (signs of overfitting)


Training & Validation (Decoder)

Metric	Value
Training Accuracy	~88%
Validation Accuracy	~83%
Confidence Score Output	Often stuck near 0.49–0.51
Misclassifications	Higher — especially on obvious spam


⚠️ Issues Observed
Overfitting between the encoder/decoder phases
The decoder struggled to generalize without a true sequence output
Inconsistent predictions (confidence always ~0.5)
Architecture was too complex for binary classification


🔁 Decision to Refactor
Realized that:
Encoder-decoder is ideal for sequence-to-sequence, not classification
LSTM classifier with sigmoid output is simpler, faster, and more accurate
Easier to train, debug, and deploy via Streamlit




### ✅ **Refined Approach: LSTM Binary Classifier**
- One Embedding + LSTM stack → Output 0 or 1
- Faster training
- Higher accuracy
- Simpler to deploy (especially in Streamlit)

---

## 🛠️ Technologies Used

| Tech         | Purpose                          |
|--------------|----------------------------------|
| Python       | Programming language             |
| TensorFlow/Keras | Deep learning framework        |
| Pandas/Numpy | Data handling                    |
| Streamlit    | Web app deployment               |
| Seaborn/Matplotlib | Visualizations               |
| Scikit-learn | Evaluation & metrics             |

---

## 🧠 Final Model Architecture

Input → Embedding → LSTM → Dropout → LSTM → Dropout → Dense (sigmoid)


- Uses tokenized tweet sequences
- Padded to fixed length (100)
- Outputs a binary prediction (0 = ham, 1 = spam)

---

## 📊 Results

| Metric              | Score   |
|---------------------|---------|
| Accuracy (Validation) | 98.3%   |
| False Positives      | 0       |
| False Negatives      | 19      |

### 📈 Training Curves

![Accuracy & Loss](images/accuracy_loss.png)

### 📉 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

## 🌐 Streamlit Web App

Interact with the model in a real-time web interface using [Streamlit](https://streamlit.io/).

### ✨ Features
- Type or paste any tweet
- Get immediate spam/ham classification
- See model confidence
- View confusion matrix and performance metrics

---

## 💻 How to Run Locally

1. Clone the repo:


2. Install dependencies:

pip install -r requirements.txt


3. Train the model (optional):

python tweetsanalysis.py


4. Run the Streamlit app:

streamlit run streamlit_app.py

