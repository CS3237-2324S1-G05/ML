import os

from ultralytics import YOLO, utils
from PIL import Image
import cv2

import paho.mqtt.client as mqtt
from time import sleep

MAIN_BROKER = "192.168.86.181"

def detectPerson(filename):
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
        connect_mqtt(toSend)


def connect_mqtt(carplate):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code: " + str(rc))
            print("Waiting for 2 seconds")
            sleep(2)

            print("Sending message.")
            client.publish("status/entrance/human-presence", carplate)
        else:
            print("Failed to connect, return code %d\n")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect("192.168.86.80", 1883, 60)
    client.loop_start()
    sleep(5)
    client.disconnect()

    return client

detectPerson('testImages/blurblur.jpeg')