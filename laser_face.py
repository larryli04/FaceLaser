import socket
import random
import time
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket successfully created")


port = 6969

s.bind((socket.gethostname(), port))
print("socket binded to %s" %(port))

s.listen(1)
print("socket is listening")

c, addr = s.accept()
print('Got connection fromm', addr)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400, height=300)
    #print(type(frame))


    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))


    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object


        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #print(type(detections))
        (startX, startY, endX, endY) = box.astype("int")
        print((startX, startY, endX, endY))

        message_x = (endX+startX)/2
        message_x = 115+(message_x/400)*(54-115)
        print(message_x)
        message_y = (endY+startY)/2
        message_y = 55+(message_y/300)*(99-53)
        print(message_y)
        message = str(message_x)[:3] + "x" + str(message_y)[:3]
        print(message, message_x, message_y)
        if(len(message)<8):
            message += " "*(8-len(message))
        message = message.replace(".", " ")
       # send a thank you message to the client.
        print(message.encode('utf-8'))
        if(not "   " in message):
            c.sendall(message.encode('utf-8'))

        time.sleep(0.05)

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
