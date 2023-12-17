from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os

class SignLanguageModel:
    def __init__(self, img_save_path='image_data', class_map=None):
        self.img_save_path = img_save_path
        self.class_map = class_map or {
            0: "NONE", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I",
            10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q",
            18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z",
        }
        self.img_size = (224, 224)
        self.num_classes = len(self.class_map)
        self.dataset = self.load_dataset()
        self.train_datagen, self.test_datagen = self.create_data_generators()
        self.training_set, self.test_set = self.create_data_flow()
        self.vgg_model = self.build_vgg_model()
        self.model = self.build_full_model()

    def load_dataset(self):
        dataset = []
        for directory in os.listdir(self.img_save_path):
            path = os.path.join(self.img_save_path, directory)
            if not os.path.isdir(path):
                continue
            for item in os.listdir(path):
                if item.startswith("."):
                    continue
                img = cv2.imread(os.path.join(path, item))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                dataset.append([img, directory])
        return dataset

    def mapper(self, val):
        return self.class_map[val]

    def create_data_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        return train_datagen, test_datagen

    def create_data_flow(self):
        data, labels = zip(*self.dataset)
        labels = list(map(self.mapper, labels))

        training_set = self.train_datagen.flow_from_directory(
            self.img_save_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical'
        )

        test_set = self.test_datagen.flow_from_directory(
            self.img_save_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical'
        )

        return training_set, test_set

    def build_vgg_model(self):
        vgg = VGG16(input_shape=[*self.img_size, 3], weights='imagenet', include_top=False)

        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        prediction = Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs=vgg.input, outputs=prediction)

    def build_full_model(self):
        model = Model(inputs=self.vgg_model.input, outputs=self.vgg_model.output)

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def train_model(self, epochs=10):
        r = self.model.fit_generator(
            self.training_set,
            validation_data=self.test_set,
            epochs=epochs,
            steps_per_epoch=len(self.training_set),
            validation_steps=len(self.test_set)
        )

    def save_model(self, model_path="sign_model"):
        self.model.save(model_path)

# Example usage:
sign_model = SignLanguageModel()
sign_model.train_model(epochs=10)
sign_model.save_model()
