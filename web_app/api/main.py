from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import base64
import io
from model import transform_image, get_prediction

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.data is None or request.data == "":
            return jsonify({'error': 'no data'})            
        try:
            imgdata = base64.decodebytes(request.data)
            tensor = transform_image(io.BytesIO(imgdata))
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

if __name__ == '__main__':
    app.run()