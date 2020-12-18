import os
import face_recognition
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

global basedir
basedir = './Datasets'
labels_filename = 'labels.csv'


class PreProcessor:
    @staticmethod
    def shape_to_np(shape, dtype='int'):
        coords = np.zeros((len(shape), 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, len(shape)):
            coords[i] = (shape[i][0], shape[i][1])

        # return the list of (x, y)-coordinates
        return coords

    def region_shape(self, img, region):
        temp_shape = []
        for reg in region:
            temp_shape += img[reg]
        n_size = len(temp_shape)
        temp_shape = np.reshape(self.shape_to_np(temp_shape), [n_size * 2])
        features = np.reshape(np.transpose(temp_shape), [n_size, 2])
        return features

    def region_extraction(self, image_paths, labels, region, extra_test):
        X = []
        Y = []
        for img_path in image_paths:
            img_name = img_path.split('.')[1].split('/')[-1]
            image = face_recognition.load_image_file(img_path)

            face_landmarks_list = face_recognition.face_landmarks(image)

            # Skip non detected faces
            if len(face_landmarks_list):
                features = self.region_shape(face_landmarks_list[0], region)

                X.append(features)
                Y.append(labels[img_name])

        X_final = np.array(X)
        Y_final = np.array(Y)

        Y = np.array([Y_final, -(Y_final - 1)]).T
        if extra_test:
            test_size, n, y = X_final.shape
            return X_final.reshape((test_size, n * y)), list(zip(*Y))[0]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_final, Y, test_size=0.2)

            train_size, n, y = X_train.shape
            test_size, _, _ = X_test.shape

            return X_train.reshape((train_size, n * y)), X_test.reshape((test_size, n * y)), list(zip(*y_train))[0], \
                   list(zip(*y_test))[0]

    def face_extraction(self, image_paths, labels, extra_test):
        X = []
        Y = []
        for img_path in image_paths:
            img_name = img_path.split('.')[1].split('/')[-1]
            image = face_recognition.load_image_file(img_path)

            face_encoding = face_recognition.face_encodings(image)

            # Skip non detected faces and multiple face images
            if len(face_encoding) == 1:
                X.append(face_encoding[0])
                Y.append(labels[img_name])

        return self.return_final_sets(X, Y, extra_test)

    def feature_extraction(self, label, images_dir, region=None, extra_test=False):
        img_dir = os.path.join(basedir, images_dir)
        label_dir = images_dir.split('/')[0]
        image_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
        labels_file = open(os.path.join(basedir, f"{label_dir}/{labels_filename}"), 'r')

        lines = labels_file.readlines()
        labels = {line.split('\t')[0]: int(line.split('\t')[label]) for line in lines[1:]}
        if region is None:
            return self.face_extraction(image_paths, labels, extra_test)
        else:
            return self.region_extraction(image_paths, labels, region, extra_test)

    def haar_cascade_feature_extraction(self, label, images_dir, extra_test):
        img_dir = os.path.join(basedir, images_dir)
        label_dir = images_dir.split('/')[0]
        image_paths = [os.path.join(img_dir, l) for l in os.listdir(img_dir)]
        labels_file = open(os.path.join(basedir, f"{label_dir}/{labels_filename}"), 'r')

        lines = labels_file.readlines()
        labels = {line.split('\t')[0]: int(line.split('\t')[label]) for line in lines[1:]}
        features = []
        eye_labels = []
        for path in image_paths:
            img_name = path.split('.')[1].split('/')[-1]
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            img = cv2.imread(path)
            eyes = haar_cascade.detectMultiScale(img, 1.3, 3)
            if len(eyes):
                x, y, w, h = eyes[0]
                cropped_img = img[y:y + h, x:x + w]
                cropped_img = cv2.resize(cropped_img, dsize=(25, 25))
                flatten_image = cropped_img.flatten()
                features.append(flatten_image)
                eye_labels.append(labels[img_name])

        return self.return_final_sets(features, eye_labels, extra_test)

    @staticmethod
    def return_final_sets(features, labels, extra_test):
        X_final = np.array(features)
        Y_final = np.array(labels)

        Y = np.array([Y_final, -(Y_final - 1)]).T
        if extra_test:
            return X_final, list(zip(*Y))[0]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_final, Y, test_size=0.2)
            return X_train, X_test, list(zip(*y_train))[0], list(zip(*y_test))[0]
