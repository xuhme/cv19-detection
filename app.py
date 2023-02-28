from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='/assets',
            static_folder='./flask app/assets',
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return render_template('upload.html')


@app.route('/upload.html')
def upload():
    return render_template('upload.html')


@app.route('/upload_chest.html')
def upload_chest():
    return render_template('upload_chest.html')


@app.route('/uploaded_chest', methods=['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

    xception_chest = load_model('models/xception_chest.h5')

    image = cv2.imread(
        './flask app/assets/images/upload_chest.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)

    xception_pred = xception_chest.predict(image)
    probability = xception_pred[0]
    print("Xception Predictions:")
    if probability[0] > 0.5:
        xception_chest_pred = str(
            '%.2f' % (probability[0]*100) + '% Positive')
    else:
        xception_chest_pred = str(
            '%.2f' % ((1-probability[0])*100) + '% Negative')
    print(xception_chest_pred)

    return render_template('results_chest.html', xception_chest_pred=xception_chest_pred)


if __name__ == '__main__':
    app.secret_key = ".."
    app.run()
