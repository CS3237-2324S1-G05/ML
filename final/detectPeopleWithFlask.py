import os

from flask import Flask, request, jsonify, url_for, redirect
from ultralytics import YOLO, utils
from PIL import Image

import paho.mqtt.client as mqtt
from time import sleep

IPADDRESS_MQTT = "192.168.86.80"
MY_IP_ADDRESS = "192.168.86.249"

TOPIC_MQTT = "status/ml/entrance/human-presence"

def connect_mqtt(status, publish_endpoint):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code: " + str(rc))
            print("Waiting for 2 seconds")
            sleep(2)

            print("Sending message.")
            client.publish(publish_endpoint, status)
        else:
            print("Failed to connect, return code %d\n")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(IPADDRESS_MQTT, 1883, 60)
    client.loop_start()
    sleep(5)
    client.disconnect()

    return client

def person_model(filename):
    photo_path = os.path.join(filename)
    photo_path_out = '{}_out.png'.format(photo_path)

    model_path = 'people.pt'

    # Load a model
    model = YOLO(model_path)  # load a custom model

    # Run inference on an image
    results = model(photo_path)  # results list
    print(results)
    # View results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image
        toSend = ""
        if len(r.boxes) == 0:
            toSend = "FALSE"
        else:
            toSend = "TRUE"
            
        connect_mqtt(toSend, TOPIC_MQTT)


app = Flask(__name__)

# Set the directory where you want to store the uploaded images
UPLOAD_FOLDER = 'people'
app.config['PEOPLE_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file is an allowed image format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/ml/human', methods=['POST'])
def upload_image():
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
        filename = os.path.join(app.config['PEOPLE_FOLDER'], file.filename)
        file.save(filename)
        person_model(filename)
        return redirect("image received")

if __name__ == '__main__':
    # Ensure the "uploads" folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host=MY_IP_ADDRESS, port=8000)