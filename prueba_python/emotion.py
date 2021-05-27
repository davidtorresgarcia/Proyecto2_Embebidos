import cv2
import numpy as np
#import tensorflow as tf
import tflite_runtime.interpreter as tflite

#Load the TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)         ######happy.png
#test_img = cv2.imread("happy.png")

while True:
    _,test_img=cap.read()# captures frame and returns boolean value and captured image
    # if not ret:
    #   continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        img_pixels = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        img_pixels = np.array(img_pixels, dtype='f')

        interpreter.set_tensor(input_details[0]['index'], img_pixels)
        interpreter.invoke()


        output_data= interpreter.get_tensor(output_details[0]['index'])
        max_index = np.argmax(output_data)

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break
    # cap.release()
# test_img.release()
cv2.destroyAllWindows
