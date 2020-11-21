import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils


def male_female_identifier(frame, faceNet, ageNet, loaded_model, minConf=0.5):
    AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    faces = []
    results = []
    predictions = []

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > minConf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            face = frame[startY:endY, startX:endX]
            f = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (300, 300))
            f = img_to_array(f)
            f = preprocess_input(f)

            faces.append(f)

            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]
            d = {
                'loc': (startX, startY, endX, endY),
                'age': (age, ageConfidence),
            }

            results.append(d)

    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        predictions = loaded_model.predict(faces, batch_size=32)

    return (results, predictions)


print('[INFO] Loading Face Detector Model...')
prototxt_path = r'face_detector\deploy.prototxt'
weights_path = r'face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxt_path, weights_path)


print('[INFO] Loading Age Detector Model...')
prototxt_path = r'age_detector\age_deploy.prototxt'
weights_path = r'age_detector\age_net.caffemodel'
ageNet = cv2.dnn.readNet(prototxt_path, weights_path)


loaded_model = load_model('Male_Female_Identifier.model')


print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    results, predictions = male_female_identifier(frame, faceNet, ageNet, loaded_model, minConf=0.5)

    for (result, prediction) in zip(results, predictions):
        (male, female) = prediction

        label = 'MALE' if male > female else 'FEMALE'
        color = (0, 255, 0) if label == 'MALE' else (255, 255, 255)

        text = '{}: {}: {:.2f}%'.format(label, result['age'][0], result['age'][1] * 100)

        (startX, startY, endX, endY) = result['loc']

        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    cv2.imshow('Male Female Identifier', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.stop()
