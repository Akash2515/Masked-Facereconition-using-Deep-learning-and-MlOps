
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer



# Define a flask app
app = Flask(__name__)

# # Model saved with Keras model.save()
MODEL_PATH = r'C:\Users\MSI GF63\Desktop\mk_Dataset\model_dl.h5'

# # Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    #  x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    img = np.vstack([x])
    classes = model.predict(img, batch_size=256)
    probabilities = model.predict_proba(img, batch_size= 256)
    probabilities_formatted = list(map("{:.2f}%".format, probabilities[0]*100))
    print(probabilities_formatted)
    return probabilities_formatted


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])

def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        person=['Akash','Rinub']
        print(f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        # print(basepath)
        
        
        print(basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # resut=str
        # if preds in person:
        #     for i in preds:
        #         result='The person name is '+ ' '+ i 
        if preds==['0.00%', '100.00%']:
            result='You are authorized. The person name is Rinub'
    
        elif preds==['100.00%', '0.00%']:
            result='You are authorized. The person name is Akash'
        else:
            result='The person is not authorized !'
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
