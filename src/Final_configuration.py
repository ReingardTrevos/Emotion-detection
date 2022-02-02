import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = base_dir + '/train'
val_dir = base_dir + '/test'

num_train = 27922
num_validation = 7965
batch_size = 64
num_epoch = 20

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


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

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['acc'])
history = model.fit(
    train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=num_epoch,
    validation_data=validation_generator,
    validation_steps=num_validation // batch_size)

file = open("parameters/all_history.txt", "a")
file.write('Evaluation:')
file.write('\n')
file.write(str(history.history.keys()))
file.write('\n')
file.write(str((history.history.values())))
file.write('\n')
file.write('\n')
file.write('Accuracy:')
file.write('\n')
file.write(str(history.history['acc']))
file.write('\n')
file.write('\n')
file.write('Val_Accuracy:')
file.write('\n')
file.write(str(history.history['val_acc']))
file.write('\n')
file.write('\n')
file.write('Loss:')
file.write('\n')
file.write(str(history.history['loss']))
file.write('\n')
file.write('\n')
file.write('Val_loss:')
file.write('\n')
file.write(str(history.history['val_loss']))
file.write('\n')
file.close()



model.save("parameters/my_model.h5")
print('\n')
