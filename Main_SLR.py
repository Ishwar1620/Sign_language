import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import Counter

class SignLanguageRecognizer:
    def __init__(self, model_path="../Downloads/sign_model.h5"):
        self.model = load_model(model_path)
        self.REV_CLASS_MAP = {
            0: "NONE",
            1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I",
            10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q",
            18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z",
        }
        self.recognized_letters = []
        self.buffer = []

    def get_key(self, val):
        occurence_count = Counter(self.recognized_letters)
        for key, value in occurence_count.items():
            if val == key:
                return value

    def mapper2(self, val):
        return self.REV_CLASS_MAP[val]

    def preprocess_frame(self, frame):
        # (You may need to adjust the region of interest and preprocessing steps)
        frame = cv2.flip(frame, 1)
        x1 = int(0.5*frame.shape[1])
        y1 = 100
        x2 = frame.shape[1]-100
        y2 = int(0.5*frame.shape[1])
        cv2.rectangle(frame, (x2+1, y2+1),(x1-1, y1-1), (255,0,0) ,1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = frame[y1:y2, x1:x2]
        img = cv2.resize(roi, (224, 224))
        img2 = img.copy()
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array ,img2

    def recognize_letter(self, frame):
        img_array ,img2= self.preprocess_frame(frame)
        pred = self.model.predict(img_array)
        pred_class = np.argmax(pred)
        move_name = self.mapper2(pred_class)
        return move_name,img2

    def run_recognition(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            move_name,img2 = self.recognize_letter(frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img2, "This is " + move_name,
                (50, 50), font, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                #cv2.imshow('frame',frame)q
            img_resized = cv2.resize(img2, (500, 500))
            cv2.imshow('frame1',img_resized)

            if move_name != "NONE":
                self.buffer.append(move_name)
            else:
                self.buffer = []

            self.recognized_letters.extend(self.buffer)

            if len(self.buffer) > 20:
                self.buffer = []

            if len(self.recognized_letters) != 0:
                most_common_letter = Counter(self.recognized_letters).most_common(1)[0][0]
                if self.get_key(most_common_letter) == 15:
                    print(most_common_letter, end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindoqws()

if __name__ == "__main__":
    recognizer = SignLanguageRecognizer()
    recognizer.run_recognition()
