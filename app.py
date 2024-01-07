from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import json
from jinja2 import Environment, FileSystemLoader

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def load_about_disease(preds):

    with open('about_disease.json') as json_file:
        data = json.load(json_file)

    # Step 2: Prepare data for rendering
    if preds == 0:
        template_data = {'citrus_canker': data.get('CitrusCanker', {})}
    elif preds == 1:
        template_data = {'citrus_greening': data.get('CitrusGreening', {})}
    elif preds == 2:
        template_data = {'citrus_gummosis': data.get('CitrusGummosis', {})}
    elif preds == 3:
        template_data = {'healthy_citrus_plant': data.get('HealthyCitrusPlant', {})}
    elif preds == 4 :
        template_data = {'leaf_miner': data.get('LeafMiner', {})}
    elif preds == 5 :
        template_data = {'lemon_butterfly': data.get('LemonButterfly', {})}
    else:
        template_data = "Not Available"
    data_about_disease = str(template_data)
    return data_about_disease

def model_predict(img_path, model):
    print(img_path)
    img = Image.open(img_path).resize((224, 224))  

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    data_about_disease = load_about_disease(preds)
    if preds==0:
        preds="The leaf is diseased with cranker\n"
    elif preds==1:
        preds="The leaf is diseased with greening\n"
    elif preds==2:
        preds="The leaf is diseased with gummosis\n"
    elif preds==3:
        preds="The leaf is healthy"
    elif preds==4:
        preds = "The leaf is diseased with leaf miner\n"
    elif preds==5:
        preds = "The leaf is diseased with lemon butterfly\n"
    else:
        preds = "Not available\n"
   
    return preds+"\n"+str(data_about_disease)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5005,debug=True)