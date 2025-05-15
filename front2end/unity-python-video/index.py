from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)


def process_frame(frame):
    # 进行边缘检测处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_frame = process_frame(frame)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': encoded_frame})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8769)
