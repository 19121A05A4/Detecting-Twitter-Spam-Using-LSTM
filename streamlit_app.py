import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load tokenizer
with open("tokenizer.json") as f:
    tokenizer = tokenizer_from_json(json.load(f))

# Load model
model = tf.keras.models.load_model("model.h5")
maxlen = 100

# UI
st.title("📢 Twitter Spam Detection")
tweet = st.text_area("Enter a tweet:")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=maxlen)
        prediction = model.predict(padded)[0][0]
        st.write(f"**Model confidence:** {prediction:.2f}")

        if prediction > 0.5:
            st.error("🚫 Spam Detected!")
        else:
            st.success("✅ This is a Ham (legit) tweet.")

