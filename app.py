from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('model_final.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    path = request.get_json(force=True)
    url = path['image']
    print(url)   
    prediction = get_output(url) 
    print(prediction)
    data = {'predict': prediction}
    return jsonify(data)

if __name__ == '__main__':
    app.run()
