from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

FOLDER = './uploads'
app.config['FOLDER'] = FOLDER

model = load_model(r'model\keras_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# imag.jpg

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['FOLDER'], filename)
        file.save(filepath)

        prediction = classify_image(filepath)
        
        return render_template('result.html', filename=filename, prediction=prediction)

    return redirect(request.url)

def classify_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_names = open("model/labels.txt", "r").readlines()
    class_label = f'Class is: {class_names[class_idx]}'

    return class_label

if __name__ == '__main__':
    app.run(debug=True)
