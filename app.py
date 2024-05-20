import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text_features = [str(x) for x in request.form.values()]
    final_features = tfidf_vectorizer.transform(text_features)
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Predicted Law/Regulation: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
