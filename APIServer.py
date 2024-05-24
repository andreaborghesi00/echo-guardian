from flask import Flask, request, send_file, jsonify
from io import BytesIO
from PIL import Image
from NNClassification import NNClassifier
from UnetSegmenter import UnetSegmenter
import sys
from flask_httpauth import HTTPBasicAuth
import hashlib
from functools import wraps

app = Flask(__name__)
auth = HTTPBasicAuth()
users = {
    "admin": hashlib.sha256("trental".encode()).hexdigest()
}

def check_auth(username, password):
    return users.get(username) == password # comes already hashed

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return jsonify({'error': 'Unauthorized access'}), 401
        print("Authorized access")
        return f(*args, **kwargs)
    return decorated

@app.route('/api/login', methods=['POST'])
@requires_auth
def login():
    return jsonify({'message': 'Successfully logged in'})

@app.route('/api/segment', methods=['POST'])
@requires_auth
def segment_image():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'Invalid image filename'}), 400

    image_bytes = image_file.read()
    image_io = BytesIO(image_bytes)
    try: image = Image.open(image_io)
    except: return jsonify({'error': 'Invalid image file'}), 400

    masked_prediction = segmenter.predict(image=image)
    
    masked_io = BytesIO()
    masked_img = Image.fromarray(masked_prediction)
    masked_img.save(masked_io, format='PNG')
    masked_io.seek(0)

    return send_file(masked_io, mimetype='image/png')

@app.route('/api/classify', methods=['POST'])
@requires_auth
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    if 'mask' not in request.files:
        return jsonify({'error': 'No mask file found'}), 400

    image_file = request.files['image']
    mask_file = request.files['mask']

    if image_file.filename == '':
        return jsonify({'error': 'Invalid image filename'}), 400

    if mask_file.filename == '':
        return jsonify({'error': 'Invalid mask filename'}), 400

    image_bytes = image_file.read()
    mask_bytes = mask_file.read()
    image_io = BytesIO(image_bytes)
    mask_io = BytesIO(mask_bytes)
    image = Image.open(image_io)
    mask = Image.open(mask_io)
    prediction = classifier.predict(image=image, mask=mask)

    return jsonify({'prediction': prediction.item()})

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python APIServer.py <path to segmenter> <path to classifier>")
        sys.exit(1)

    segmenter_path = sys.argv[1]
    classifier_path = sys.argv[2]

    print("Loading Vision Transformer model...")
    segmenter = UnetSegmenter(model_path=segmenter_path)
    print("Model loaded successfully!")

    print("Loading ImageNet class labels...")
    classifier = NNClassifier(model_path=classifier_path)
    print("Class labels loaded successfully!")

    print("-" * 50)
    print("Starting API server...")
    app.run()