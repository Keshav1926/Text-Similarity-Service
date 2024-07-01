import spacy
from flask import Flask, request, jsonify
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('word_tokenize')
application = Flask(__name__)
nlp = spacy.load('en_core_web_md')

def preprocess_text(text):
    text = text.lower()
    
    text = re.sub(r"[^a-zA-Z]", " ", text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    preprocessed_text = " ".join(filtered_tokens)
    
    return preprocessed_text

def calculate_similarity(text1, text2):
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    doc1 = nlp(preprocessed_text1)
    doc2 = nlp(preprocessed_text2)
    similarity_score = doc1.similarity(doc2)
    return similarity_score

@application.route("/calculate_similarity/", methods=["POST"])
def get_similarity_score():
    request_data = request.get_json()
    text1 = request_data.get("text1", "")
    text2 = request_data.get("text2", "")

    if not text1 or not text2:
        return jsonify({"error": "Both 'text1' and 'text2' fields are required."}), 400

    similarity_score = calculate_similarity(text1, text2)
    return jsonify({"similarity score": similarity_score})

if __name__ == "__main__":
    application.run(debug=True, host=0.0.0.0 , port = 80)

