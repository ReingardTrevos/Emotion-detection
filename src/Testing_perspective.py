import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1183991828928715))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.21206140798448064))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.20158693285958107))
model.add(Dense(7, activation='softmax'))

model.load_weights('best_my_model')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   GNIEW   ", 1: "  WSTRET  ", 2: "  STRACH  ", 3: "SZCZESCIE", 4: "NEUTRALNOSC", 5: "   SMUTEK   ", 6: "ZASKOCZENIE"}

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(150, 120),
          font_scale=2,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

i = 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        #cv2.rectangle(frame, (x-50, y+170), (x + w, y + h), (0, 0, 0), -1)
        #cv2.putText(frame, emotion_dict[maxindex], (x-50, y+170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        draw_text(frame, emotion_dict[maxindex], pos=(x-60, y-70))

    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite('kang' + str(i) + '.jpg', frame)
        i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
