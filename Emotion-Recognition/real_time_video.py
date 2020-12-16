from keras.preprocessing.image import img_to_array
import imutils
import cv2

from tensorflow.keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/generated_model/model9.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), " Physical GPUs, ", len(logical_gpus), " Logical GPUs")

    except RuntimeError as e:
        print(e)

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]


# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (96, 96)) #original - 48, 48
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]

        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        emoji_img = cv2.imread("./emoji/{}.png".format(label), -1)
        emoji_img = cv2.resize(emoji_img, (fW, fH))

        alpha = emoji_img[:, :, 3]/255.0

        frameClone[fY:fY + fH, fX:fX + fW, 0] = frameClone[fY:fY + fH, fX:fX + fW, 0]*(1-alpha) +alpha*emoji_img[:,:,0]
        frameClone[fY:fY + fH, fX:fX + fW, 1] = frameClone[fY:fY + fH, fX:fX + fW, 1] * (1 - alpha) + alpha * emoji_img[:, :, 1]
        frameClone[fY:fY + fH, fX:fX + fW, 2] = frameClone[fY:fY + fH, fX:fX + fW, 2] * (1 - alpha) + alpha * emoji_img[:, :, 2]


    else: continue

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)

        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)


    frameClone = cv2.resize(frameClone, (1024, 1000))
    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



camera.release()
cv2.destroyAllWindows()
