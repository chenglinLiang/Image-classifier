from __future__ import print_function
import json
import numpy as np
import requests
import os
from flask import Flask, render_template
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_wtf import FlaskForm
from wtforms import SubmitField
import uuid
from keras.preprocessing import image
from keras.applications import inception_v3

app = Flask(__name__)
app.secret_key = 'secret string'
app.config['UPLOAD_PATH'] = os.path.join(app.root_path, 'upload')


class UploadForm(FlaskForm):
    image = FileField('Upload Image', validators=[FileRequired(), FileAllowed(['jpeg', 'jpg', 'png', 'gif'])])
    submit = SubmitField()


def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    list = [[0,0],[0,0],[0,0]]
    if form.validate_on_submit():
        f = form.image.data
        filename = random_filename(f.filename)
        image_path = os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(image_path)
        img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.
        payload = {
            "instances": [{'input_image': img.tolist()}]
        }
        r = requests.post('http://localhost:8501/v1/models/InceptionV3:predict', json=payload)
        pred = json.loads(r.content.decode('utf-8'))
        preds = (inception_v3.decode_predictions(np.array(pred['predictions']))[0])[:3]
        list = []
        for pred in preds:
            pred = [pred[1],pred[2]*100]
            list.append(pred)
    return render_template('index.html', list=list, form=form)

@app.route('/about/')
def about():
    return render_template('about.html')
