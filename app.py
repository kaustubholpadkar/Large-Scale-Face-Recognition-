import cv2
import numpy as np
from cache import lru
from flask import Flask, request, Response
from face_recognizer import FaceRecognizer

cache = lru.LRUCache(capacity=10)
face_recognizer = FaceRecognizer(cache=cache, threshold=0.7, embedding_dir="./models")

app = Flask(__name__)


@app.route('/api/recognize', methods=['POST'])
def recognize():
    file = request.files['face']
    data = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    identity = request.form['identity']
    print(identity)
    identity = face_recognizer.recognize(img, identity)
    print(identity)
    return Response(response=identity, status=200)


# start flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
