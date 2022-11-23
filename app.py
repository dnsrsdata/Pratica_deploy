from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    predictions = model.predict(final_features)
    return render_template('index.html', prediction_text = 'O estoque está {}'.format(predictions))
    

if __name__ == '__main__':
    app.run(debug = True)