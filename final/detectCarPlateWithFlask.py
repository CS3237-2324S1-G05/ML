#source for carplate recognition: https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8/blob/main/util.py
#dataset for carplate recognition: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4

from paddleocr import PaddleOCR
from PIL import Image
import string
import os
import paho.mqtt.client as mqtt
from time import sleep
from flask import Flask, request, jsonify, redirect
from ultralytics import YOLO

IPADDRESS_MQTT = "192.168.43.186"
MY_IP_ADDRESS = "192.168.43.28"

TOPIC_MQTT_ENTRANCE = "status/ml/entrance/carplate"
TOPIC_MQTT_EXIT = "status/ml/exit/carplate"
TOPIC_MQTT_LOT = "status/ml/lot/carplate/"

# Initialize the OCR reader
reader = PaddleOCR(use_angle_cls=True, lang='en')

# Mapping dictionaries for character conversion
dict_first_letter = {'S': 'S', '5': 'S'}

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'B': '8',
                    'S': '5'}

dict_int_to_char = {'0': 'D',
                    '2': 'Z',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '7': 'Z',
                    '8': 'B',
                    '@': 'D'}
    
def format_carplate(text):

    plate = ''

    if text[0] == '[':
        text = text[1:]

    map = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int,
                2: dict_int_to_char, 3: dict_char_to_int, 7: dict_int_to_char}
    for j in [0, 1, 2, 3, 4, 5, 6, 7]:
        if text[j] in map[j].keys():
            plate += map[j][text[j]]
        else:
            plate += text[j]

    return plate

def correct_carplate_format(text):
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

def read_carplate(carplate_crop):
    detections = reader.ocr(carplate_crop)
    result = 'INVALID'

    for detection in range(len(detections)):
        res = detections[detection]
        for line in res:
            carplate_crop_str = str(line[-1][0]).replace(" ", "")
            if (correct_carplate_format(carplate_crop_str)):
                result = format_carplate(carplate_crop_str)
    
    return str(result)

#source: https://stackoverflow.com/questions/76899615/yolov8-how-to-save-the-output-of-model
def detect_carplate(carplate_img, model_path, mqttStr):
    photo_path = os.path.join(carplate_img)
    img_raw = Image.open(photo_path)
    
    # Load model
    model = YOLO(model_path)  
    
    # Run inference
    detected = model(img_raw)
    result = ''    
    carplate_boxes = detected[0].boxes.data.cpu().numpy()
    print(f"License Plate Detected: {detected[0].boxes.data.cpu().numpy()}")

    if detected[0].boxes.data.cpu().numpy().size != 0: #carplate detected
        for i, box in enumerate(carplate_boxes): #carplate detected
            x1, y1, x2, y2, conf, cls = box
        carplate = img_raw.crop((x1, y1, x2, y2))
        carplate_filename = f'plates/carplate_{i+1}.jpg'
        carplate.save(carplate_filename)

        result = read_carplate(carplate_filename)
    
    print(f"License Plate number: {result}")
    connect_mqtt(result, mqttStr)

    return result

def connect_mqtt(carplate, publish_endpoint):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code: " + str(rc))
            print("Waiting for 2 seconds")
            sleep(2)

            print("Sending message.")
            print(carplate)
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

# Set image directory
UPLOAD_FOLDER_ENTRANCE = 'entrance'
app.config['CARS_FOLDER_ENTRANCE'] = UPLOAD_FOLDER_ENTRANCE
UPLOAD_FOLDER_CARPARK = 'carpark'
app.config['CARS_FOLDER_CARPARK'] = UPLOAD_FOLDER_CARPARK
UPLOAD_FOLDER_EXIT = 'exit'
app.config['CARS_FOLDER_EXIT'] = UPLOAD_FOLDER_EXIT

# Check if allowed format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/ml/entrance', methods=['POST'])
def upload_image_entrance():
    flask_body('CARS_FOLDER_ENTRANCE', TOPIC_MQTT_ENTRANCE)

@app.route('/ml/carpark', methods=['POST'])
def upload_image_carpark():

    id = request.headers.get('id')
    print(id)
    mqttStr = TOPIC_MQTT_LOT + "/" + id
    flask_body('CARS_FOLDER_CARPARK', mqttStr)
    
@app.route('/ml/exit', methods=['POST'])
def upload_image_exit():
    flask_body('CARS_FOLDER_EXIT', TOPIC_MQTT_EXIT)

def flask_body(upload_string, mqttStr):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    if file:
        filename = os.path.join(app.config[upload_string], file.filename)
        file.save(filename)
        detect_carplate(filename, 'last.pt', mqttStr)
        return redirect("message received")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER_ENTRANCE):
        os.makedirs(UPLOAD_FOLDER_ENTRANCE)

    if not os.path.exists(UPLOAD_FOLDER_CARPARK):
        os.makedirs(UPLOAD_FOLDER_CARPARK)

    if not os.path.exists(UPLOAD_FOLDER_EXIT):
        os.makedirs(UPLOAD_FOLDER_EXIT)

    app.run(host=MY_IP_ADDRESS, port=8080)