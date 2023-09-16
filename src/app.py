from flask import Flask, request, jsonify
import cv2
import numpy as np
from main import handwritten_text_recognition

app = Flask(__name__)

@app.route('/htr', methods=['POST'])
def main():
    if 'image' not in request.files:
        return 'No file part'


    # Extract image from request
    file = request.files['image']
    image_data = file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    output = handwritten_text_recognition(img, debug=True)

    temp_obj = {
        "ic": "31081960101984",
        "firstName": "Jerry",
        "lastName": "Seinfeld",
        "dob": "1960-08-31",
        "gender": "Male",
        "nationality": "American",
        "phoneNo": " 5558383",
        "emergencyNo": "601170735766",
        "emergencyRemarks": "You know George",
        "remarks": "What's the deal with arthritis?"
    }

    return jsonify(temp_obj)


if __name__ == "__main__":
    app.run(debug=True)