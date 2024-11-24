######## Mock-Up application for inference ########
# Prerequisite: Please download the saved distillbert model from below link and place it in 'distilbert' folder as large file cannot be saved in github
# URL: https://entuedu-my.sharepoint.com/:f:/g/personal/c004_e_ntu_edu_sg/EoDbpIa-QDVPtVGoSmcmIVEB5njF8Siu9n5-KwQ2QjtbJQ?e=dImVZ0

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from PIL import Image

# Load movie data
movie_data = {
    "movieid": "597-titanic",
    "movie_title": "Titanic",
    "movie_year": 1997,
    "movie_release_date": "19/12/1997",
    "movie_genre": "Drama|Romance|History",
    "movie_rating": 88,
    "movie_desc": "A love story unfolds on the ill-fated RMS Titanic between a young aristocrat and a poor artist.",
    "movie_cast": "Leonardo DiCaprio|Kate Winslet|Billy Zane|Kathy Bates|Bill Paxton",
    "movie_tag": "romantic|tragic|ship|disaster|true story",
    "budget": 200000000,
    "revenue": 2200000000
}

# Paths
# model_path = r"C:\Users\ceffendy\Documents\GitHub\KM6312_MovieReviewSentiment\movie_review_app\fine_tuned_model"
model_path = r"distilbert"

positive_image = Image.open("data/image/titanic_positive.jpg")
negative_image = Image.open("data/image/titanic_negative.jpg")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
# model = DistilBertForSequenceClassification.from_pretrained(model_path, from_safetensors=True).to(device)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model.eval()

# Helper functions
def preprocess_text(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    return {key: value.to(device) for key, value in encoding.items()}

def predict_sentiment(text):
    encoding = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
        positive_prob = probabilities[0][1].item()
        negative_prob = probabilities[0][0].item()
    return sentiment, positive_prob, negative_prob

# App title and style
st.set_page_config(page_title="TMDB Movie Review with Sentiment Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #1c1c1c;
            color: #f0f0f0;
            font-family: 'Roboto', sans-serif;
        }
        .card {
            background-color: #0a1f44;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #ffcc00;
        }
        h1, h2, h3 {
            color: #ffcc00;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #ffcc00;
        }
    </style>
""", unsafe_allow_html=True)

# Main layout
st.title("🎬 TMDB Movie Review with Sentiment Analysis")

# Columns for layout
col1, col2 = st.columns([1, 2])

# Movie details on the left
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/en/1/18/Titanic_%281997_film%29_poster.png", width=300)
    st.markdown(f"**Title:** {movie_data['movie_title']} ({movie_data['movie_year']})")
    st.markdown(f"**Release Date:** {movie_data['movie_release_date']}")
    st.markdown(f"**Genre:** {movie_data['movie_genre']}")
    st.markdown(f"**Rating:** {movie_data['movie_rating']}/100")
    st.markdown(f"**Description:** {movie_data['movie_desc']}")
    st.markdown(f"**Cast:** {movie_data['movie_cast']}")
    st.markdown(f"**Tags:** {movie_data['movie_tag']}")
    st.markdown(f"**Budget:** ${movie_data['budget']:,}")
    st.markdown(f"**Revenue:** ${movie_data['revenue']:,}")
    st.markdown("</div>", unsafe_allow_html=True)

# Review and sentiment analysis on the right
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Submit Your Review")
    review_text = st.text_area("Enter your review here:")

    if st.button("Analyze Review"):
        if review_text.strip():
            sentiment, positive_prob, negative_prob = predict_sentiment(review_text)

            # Create columns for sentiment analysis result and image
            result_col, image_col = st.columns([1, 1])

            # Display sentiment and probabilities
            with result_col:
                st.markdown(f"**Predicted Sentiment:** {sentiment}")
                st.markdown(f"**Positive Probability:** {positive_prob:.2f}")
                st.markdown(f"**Negative Probability:** {negative_prob:.2f}")

            # Display image based on prediction
            with image_col:
                if sentiment == "POSITIVE":
                    st.image(positive_image, caption="Positive Response", width=200)
                else:
                    st.image(negative_image, caption="Negative Response", width=200)
        else:
            st.warning("Please enter a review before analyzing.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2024 TMDB Movie Review with Sentiment Analysis. All rights reserved.</div>", unsafe_allow_html=True)