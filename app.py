# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# from flask import Flask, render_template, request
# import re
# import pickle
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords

# # Initialize Flask app
# app = Flask(__name__, template_folder='./templates', static_folder='./static')

# # Load trained model and vectorizer
# loaded_model = pickle.load(open("model.pkl", 'rb'))
# vectorizer = pickle.load(open("vector.pkl", 'rb'))

# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# def fake_news_det(news):
#     # Preprocess
#     review = re.sub(r'[^a-zA-Z\s]', '', news)
#     review = review.lower()
#     review = nltk.word_tokenize(review)
    
#     corpus = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
#     input_data = [' '.join(corpus)]
    
#     # Vectorize
#     vectorized_input_data = vectorizer.transform(input_data)
    
#     # Predict
#     prediction = loaded_model.predict(vectorized_input_data)
#     return prediction

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         message = request.form['news']
#         pred = fake_news_det(message)
        
#         if pred[0] == 1:
#             result = "Prediction of the News : ⚠ Fake News 📰"
#         else:
#             result = "Prediction of the News : ✅ Real News 📰"
        
#         return render_template("prediction.html", prediction_text=result)
#     return render_template('prediction.html', prediction="Something went wrong")

# if __name__ == '__main__':
#     app.run(debug=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from flask import Flask, render_template, request
import re
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load trained model and vectorizer
loaded_model = pickle.load(open("model.pkl", 'rb'))
vectorizer = pickle.load(open("vector.pkl", 'rb'))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def fake_news_det(news):
    # Preprocess
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)
    
    corpus = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    input_data = [' '.join(corpus)]
    
    # Vectorize
    vectorized_input_data = vectorizer.transform(input_data)
    
    # Predict
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     message = request.form['news']
#     pred = fake_news_det(message)
    
#     if pred[0] == 1:
#         result = "⚠ Fake News 📰"
#     else:
#         result = "✅ Real News 📰"
    
#     return render_template("prediction.html", prediction_text=result)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        if pred[0] == 1:
            result = "⚠ Fake News 📰"
        else:
            result = "✅ Real News 📰"
        return render_template("prediction.html", prediction_text=result)

    # if GET request, just show empty prediction.html
    return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=True)
