from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model_iris.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ip_feat = [float(x) for x in request.form.values()]
    final_feat = [np.array(ip_feat)]
    prediction = model.predict(final_feat)
    output = prediction[0]
    print(output)
    return render_template('index.html', prediction_text = 'Prediction is '+ output)

if __name__ == '__main__':
    app.run(debug = True)