#source: https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8/blob/main/util.py
#dataset: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4

import easyocr
import string
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from ultralytics import YOLO

import paho.mqtt.client as mqtt
from time import sleep

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)
results = ''
# Mapping dictionaries for character conversion
dict_first_letter = {'S': 'S', '5': 'S'}

dict_char_to_int = {'O': '0',
                    'I': '1',
                    ']': '3',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'B': '8',
                    'S': '5'}

dict_int_to_char = {'2': 'Z',
                    '7': 'Z',
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

    result = ''
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        result = format_license(text)

    return result

def connect_mqtt(carplate):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code: " + str(rc))
            print("Waiting for 2 seconds")
            sleep(2)

            print("Sending message.")
            client.publish("status/lot/carplate/0", carplate)
        else:
            print("Failed to connect, return code %d\n")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect("192.168.86.80", 1883, 60)
    client.loop_start()
    sleep(5)
    client.disconnect()

    return client
    

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
        plate_filename = f'plates/license_plate_{i+1}.jpeg'
        license_plate.save(plate_filename)

        #sharpen image
        imgCv = cv2.imread(plate_filename)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(imgCv, -1, kernel)
        blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
        cv2.imwrite(f'plates/license_plate_{i+1}.jpeg', blurred)
        
        results = str(read_license_plate(plate_filename)) #read_license_plate(plate_filename)
        print(f"License Plate number: {results}")
        connect_mqtt(str(results))

    return 0

# load the image and resize it

detect_license_plate('testImages/blurnumberplate.jpeg', 'last.pt')