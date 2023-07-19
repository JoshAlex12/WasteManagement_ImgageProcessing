from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your TensorFlow model here
model = tf.keras.models.load_model('model_final.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    if request.method == 'POST':
        # Get the uploaded image from the request
        image_file = request.files['image']
        
        # Preprocess the image (resize, normalization, etc.) before feeding it to the model
        # Example:
        image = preprocess_image(image_file)

        # Make the prediction using your TensorFlow model
        prediction = model.predict(image)
        class_label = np.argmax(prediction)

        # Customize the class labels based on your model output
        class_labels = ['Non-biodegradable Waste', 'Biodegradable Waste']
        prediction_result = class_labels[class_label]

        return jsonify({'predict': prediction_result})
    else:
        return render_template('index.html')

def preprocess_image(image_file):
    # Implement your image preprocessing logic here
    # For example, you can use TensorFlow's ImageDataGenerator or PIL library
    '''
    path = request.get_json(force=True)
    url = path['image']
    print(url)   
    prediction = get_output(url) 
    print(prediction)
    data = {'predict': prediction}
    return jsonify(data)

if __name__ == '__main__':
    app.run()
