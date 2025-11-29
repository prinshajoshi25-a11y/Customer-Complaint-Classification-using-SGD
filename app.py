import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from flask import Flask, render_template, request
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("complaint_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Initialize text cleaning tools
stopword = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopword]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

# Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    complaint_text = ""
    confidence = None  # NEW

    if request.method == 'POST':
        text = request.form.get('complaint', '').strip()
        complaint_text = text

        if not text:
            prediction_text = "⚠️ Please enter a complaint!"
        else:
            try:
                cleaned_text = clean(text)
                data_input = tfidf.transform([cleaned_text])

                # Predict probabilities
                probs = model.predict_proba(data_input)[0]
                max_prob = probs.max()
                confidence = round(max_prob * 100, 2)  # NEW: confidence in %

                threshold = 0.4

                if max_prob < threshold:
                    prediction_text = "❌ Please enter a valid complaint!"
                else:
                    output = model.classes_[probs.argmax()]
                    prediction_text = f"✅ Predicted Category: {output} "

            except Exception as e:
                prediction_text = f"❌ Error: {str(e)}"

    return render_template('index.html',
                           prediction_text=prediction_text,
                           complaint_text=complaint_text,
                           confidence=confidence)  # NEW

if __name__ == "__main__":
    app.run(debug=True)
