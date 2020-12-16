import pandas as pd
import cv2
import numpy as np


from keras.preprocessing.image import ImageDataGenerator

dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(94, 94)  # (48,48)

def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48  #48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        #faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def augmentation(x, y):
    aug_x = x.reshape(-1, 94, 94, 1)   #48, 48

    img_generator = ImageDataGenerator(rotation_range=10, zoom_range=0.10,
                                       shear_range=0.5, width_shift_range=0.10,
                                       height_shift_range=0.10, horizontal_flip=True,
                                       vertical_flip=False)

    augment_size = 10000
    # randint -> np.random.choice() 중복 방지 . replace = False
    #randidx = np.random.randint(x.shape[0], size=augment_size)
    randidx = np.random.choice(x.shape[0], size=augment_size, replace=False)
    x_augmented = x[randidx].copy()
    y_augmented = y[randidx].copy()
    x_augmented = img_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
    x = np.concatenate((x, x_augmented))
    y = np.concatenate((y, y_augmented))
    return x, y


