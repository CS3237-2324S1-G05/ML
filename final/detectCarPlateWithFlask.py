#source: https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8/blob/main/util.py
#dataset: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4

import easyocr
import string
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import paho.mqtt.client as mqtt
from time import sleep

from flask import Flask, request, jsonify, url_for, redirect
from ultralytics import YOLO

IPADDRESS_MQTT = "192.168.86.181"
MY_IP_ADDRESS = "192.168.86.249"

TOPIC_MQTT_ENTRANCE = "status/ml/entrance/carplate"
TOPIC_MQTT_EXIT = "status/ml/exit/carplate"
TOPIC_MQTT_LOT = "status/ml/lot/carplate/"

def on_connect(client, userdata, flags, rc, carplate):
    print("Connected with result code: " + str(rc))
    print("Waiting for 2 seconds")
    sleep(2)

    print("Sending message.")
    client.publish("detect/people", carplate)

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_first_letter = {'S': 'S', '5': 'S'}

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'B': '8',
                    'S': '5'}

dict_int_to_char = {'2': 'Z',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B'}

def license_complies_format(text):
    if len(text) != 8:
        return False

    if (text[0] in string.ascii_uppercase and text[0] in ['S', '5']) and \
       ((text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and text[1] not in ['1', '0', 'O', 'I']) and \
       ((text[2] in  string.ascii_uppercase or text[2] in dict_char_to_int.keys()) and text[2] not in ['1', '0', 'O', 'I'] ) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
       ((text[7] in string.ascii_uppercase or text[7] in dict_int_to_char.keys()) and text[7] not in ['F', 'I', 'N', 'O', 'V', 'W']):
        return True
    else:
        return False
    
def format_license(text):

    license_plate_ = ''

    if text[0] == '[':
        text = text[1:]

    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int,
                2: dict_int_to_char, 3: dict_char_to_int, 7: dict_int_to_char}
    for j in [0, 1, 2, 3, 4, 5, 6, 7]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        return format_license(text)

    return format_license(text)

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

#source: https://stackoverflow.com/questions/76899615/yolov8-how-to-save-the-output-of-model
def detect_license_plate(license_plate_img, model_path):
    photo_path = os.path.join(license_plate_img)
    img = Image.open(photo_path)
    
    # Load a model
    model = YOLO(model_path)  # load a custom model
    
    # Run inference on an image
    detected = model(img)
    license_plate_boxes = detected[0].boxes.data.cpu().numpy()
    print(detected[0].boxes.data.cpu().numpy())
    for i, box in enumerate(license_plate_boxes):
        x1, y1, x2, y2, conf, cls = box
        license_plate = img.crop((x1, y1, x2, y2))
        plate_filename = f'plates/license_plate_{i+1}.jpg'
        license_plate.save(plate_filename)

         #sharpen image
        img = cv2.imread(plate_filename)
        kernel = np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]])
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        auto_edge = auto_canny(blurred)
        cv2.imshow("image", auto_edge)
        cv2.waitKey(0)
        cv2.imwrite(f'plates/license_plate_{i+1}.jpg', auto_edge)

        results = read_license_plate(plate_filename)
        print(f"License Plate number: {results}")

    return 0

def connect_mqtt(carplate, publish_endpoint):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code: " + str(rc))
            print("Waiting for 2 seconds")
            sleep(2)

            print("Sending message.")
            client.publish(publish_endpoint, carplate)
        else:
            print("Failed to connect, return code %d\n")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(IPADDRESS_MQTT, 1883, 60)
    client.loop_start()
    sleep(5)
    client.disconnect()

    return client
    

app = Flask(__name__)

# Set the directory where you want to store the uploaded images
UPLOAD_FOLDER_ENTRANCE = 'entrance'
app.config['CARS_FOLDER_ENTRANCE'] = UPLOAD_FOLDER_ENTRANCE
UPLOAD_FOLDER_CARPARK = 'carpark'
app.config['CARS_FOLDER_CARPARK'] = UPLOAD_FOLDER_CARPARK
UPLOAD_FOLDER_EXIT = 'exit'
app.config['CARS_FOLDER_EXIT'] = UPLOAD_FOLDER_EXIT

# Function to check if the file is an allowed image format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/ml/entrance', methods=['POST'])
def upload_image_entrance():
    flask_body('CARS_FOLDER_ENTRANCE', TOPIC_MQTT_ENTRANCE)

@app.route('/ml/carpark', methods=['POST'])
def upload_image_carpark():
    data_dict = request.get_json()
    mqttStr = TOPIC_MQTT_LOT + data_dict
    flask_body('CARS_FOLDER_CARPARK', mqttStr)
    
@app.route('/ml/exit', methods=['POST'])
def upload_image_exit():
    flask_body('CARS_FOLDER_EXIT', TOPIC_MQTT_EXIT)

def flask_body(upload_string, mqttStr):
    # Check if a file was included in the POST request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file name is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is allowed (you can customize this function)
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    # If the file is valid, save it to the specified upload folder
    if file:
        filename = os.path.join(app.config[upload_string], file.filename)
        file.save(filename)
        result = str(detect_license_plate(filename, 'last.pt'))
        connect_mqtt(result, mqttStr)
        return redirect("message received")#url_for('upload_image_entrance', filename=filename))

if __name__ == '__main__':
    # Ensure the "uploads" folder exists
    if not os.path.exists(UPLOAD_FOLDER_ENTRANCE):
        os.makedirs(UPLOAD_FOLDER_ENTRANCE)

    if not os.path.exists(UPLOAD_FOLDER_CARPARK):
        os.makedirs(UPLOAD_FOLDER_CARPARK)

    if not os.path.exists(UPLOAD_FOLDER_EXIT):
        os.makedirs(UPLOAD_FOLDER_EXIT)

    app.run(host=MY_IP_ADDRESS, port=8080)