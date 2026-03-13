import streamlit as st
import pickle

# Page settings
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# Load model
model = pickle.load(open("model/fake_review_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Title
st.title("🕵️ Fake Product Review Detection System")

st.write(
"""
This AI model analyzes product reviews and predicts whether they are **Fake** or **Genuine**.
"""
)

st.markdown("---")

# Input review
review = st.text_area("✍️ Enter a product review")

if st.button("🔍 Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)
        probability = model.predict_proba(review_vector)

        fake_prob = probability[0][0]
        real_prob = probability[0][1]

        st.subheader("Result")

        if prediction[0] == 0:
            st.error("❌ Fake Review Detected")
            confidence = fake_prob
        else:
            st.success("✅ Genuine Review")
            confidence = real_prob

        # Confidence score
        st.write("### AI Confidence Score")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        # Review analytics
        st.write("### Review Analysis")

        word_count = len(review.split())
        st.write(f"Word Count: **{word_count} words**")

        if word_count < 5:
            st.warning("Very short reviews are often suspicious.")

st.markdown("---")
st.caption("Fake Product Review Detection using Machine Learning")