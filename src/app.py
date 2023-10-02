from flask import Flask, CORS, request, jsonify
import cv2
import numpy as np
from main import patient_registration_parser

app = Flask(__name__)
CORS(app)

@app.route('/patient_htr', methods=['POST'])
def patient_htr():
    if 'image' not in request.files:
        return 'No image'

    # Extract image from request
    file = request.files['image']
    image_data = file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    output = patient_registration_parser(img, model="keras_ocr", debug=False)

    return jsonify(output)


@app.route('/patient_htr_sample', methods=['POST'])
def patient_htr_sample():
    if 'image' not in request.files:
        return 'No image'

    temp_obj = {
        "ic": "31081960101984",
        "firstName": "Jerry",
        "lastName": "Seinfeld",
        "dob": "1960-08-31",
        "gender": "Male",
        "nationality": "American",
        "phoneNo": " 5558383",
        "email": "jerryseinfeld@gmail.com",
        "emergencyNo": "601170735766",
        "emergencyRemarks": "You know George",
        "remarks": "What's the deal with arthritis?"
    }

    return jsonify(temp_obj)



if __name__ == "__main__":
    app.run(debug=True)