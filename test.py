import cv2
from cvzone.HandTrackingModule import HandDetector  # type: ignore
import numpy as np
import math
import tensorflow as tf

cap = cv2.VideoCapture(0)  # 0 is the id number of webcam
detector = HandDetector(maxHands=2)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="Model/model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("Model/labels1.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

offset = 20
imageSize = 300

while True:
    success, img = cap.read()
    imageOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imageCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imageCropShape = imageCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)

            imageResize = cv2.resize(imageCrop, (wCal, imageSize))
            imageResizeShape = imageResize.shape
            wGap = math.ceil((imageSize - wCal) / 2)
            imageWhite[:, wGap:wCal + wGap] = imageResize

        else:
            k = imageSize / w
            hCal = math.ceil(k * h)

            imageResize = cv2.resize(imageCrop, (imageSize, hCal))
            imageResizeShape = imageResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imageWhite[hGap:hCal + hGap, :] = imageResize

        # Resize to model input size (usually 224x224)
        imgInput = cv2.resize(imageWhite, (224, 224))
        imgInput = imgInput.astype(np.float32) / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], imgInput)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        index = int(np.argmax(prediction))
        print(prediction, index)

        # Display prediction on screen
        cv2.rectangle(imageOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imageOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv2.rectangle(imageOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # cv2.imshow("ImageCrop", imageCrop)
        # cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("image", imageOutput)
    cv2.waitKey(1)
