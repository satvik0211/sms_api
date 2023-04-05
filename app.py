from flask import Flask,request,jsonify
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Its Working"

@app.route('/predict',methods =['POST'])
def predict():
    sms = request.form.get('sms')

    transformed_sms = transform_text(sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    return jsonify({'predict':str(result)})

if __name__ == '__main__':
    app.run(host ="0.0.0.0",port = 5000, debug=True)