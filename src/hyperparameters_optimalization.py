import sys
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle


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

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['model']

space = {'choice': hp.choice('num_layers',[ {'layers':'two', },
                    {'layers':'three',
                    'n23_filters_conv': hp.choice('n23_filters_conv', [16, 32, 64]),
                    '23_dropout': hp.uniform('23_dropout', 0.01, 0.99)}
                    ]),
    'n1_filters_conv': hp.choice('n1_filters_conv', [16, 32, 64]),
    'n2_filters_conv': hp.choice('n2_filters_conv', [16, 32, 64]),
    '1_dropout': hp.uniform('1_dropout', 0.01, 0.99),

    'n3_filters_conv': hp.choice('n3_filters_conv', [16, 32, 64]),
    'n4_filters_conv': hp.choice('n4_filters_conv', [16, 32, 64]),
    '2_dropout': hp.uniform('2_dropout', 0.01, 0.99),

    '1_neurons_dense': hp.choice('1_neurons_dense', [256, 512, 1024]),
    '3_dropout': hp.uniform('3_dropout', 0.01, 0.99),
}



def mi_cnn(pars):
    print('Parameters: ', pars)

    model = Sequential()
    model.add(Conv2D(pars['n1_filters_conv'], kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(pars['n2_filters_conv'], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(pars['1_dropout']))

    if pars['choice']['layers'] == 'three':
        model.add(Conv2D(pars['choice']['n23_filters_conv'], kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(pars['choice']['23_dropout']))

    model.add(Conv2D(pars['n3_filters_conv'], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(pars['n4_filters_conv'], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(pars['2_dropout']))

    model.add(Flatten())
    model.add(Dense(pars['1_neurons_dense'], activation='relu'))
    model.add(Dropout(pars['3_dropout']))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['acc'])
    history = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_validation // batch_size)

    best_epoch_loss = np.argmin(history.history['val_loss'])
    best_val_loss = np.min(history.history['val_loss'])
    best_val_acc = np.max(history.history['val_acc'])

    file = open("parameters/all_history.txt", "a")
    file.write('Evaluation:')
    file.write('\n')
    file.write(str(history.history.keys()))
    file.write('\n')
    file.write(str((history.history.values())))
    file.write('\n')
    file.write('Parameters: ')
    file.write(str(pars))
    file.write('\n')
    file.write('\n')
    file.close()

    print('Epoch {} - val acc: {} - val loss: {}'.format(best_epoch_loss, best_val_acc, best_val_loss))
    sys.stdout.flush()

    return {'loss': best_val_loss, 'best_epoch': best_epoch_loss, 'eval_time': time.time(), 'status': STATUS_OK,
            'model': model, 'history': history}
#trials = Trials()
trials = pickle.load(open("my_trials3.p", "rb"))


best = fmin(mi_cnn, space, algo=tpe.suggest, max_evals=100, trials=trials)
model = getBestModelfromTrials(trials)
pickle.dump(trials, open("parameters/my_trials4.p", "wb"))
model.save("parameters/my_model")
print('\n')
print(best)


file = open("parameters/best_best.txt", "w")
file.write('Best:')
file.write(str(best))
file.write('\n')
file.write('\n')
file.close()

file = open("parameters/trials.txt", "w")
file.write('Trails:')
file.write('\n')
file.write(str(trials.trials))
file.write('\n')
file.write('\n')
file.close()

file = open("parameters/results.txt", "w")
file.write('Results:')
file.write('\n')
file.write(str(trials.results))
file.write('\n')
file.write('\n')
file.close()

file = open("parameters/trial_argmin.txt", "w")
file.write('Trial_argmnin:')
file.write('\n')
file.write(str(trials.argmin))
file.write('\n')
file.write('\n')
file.close()

file = open("parameters/best_trial.txt", "w")
file.write('Best trails:')
file.write('\n')
file.write(str(trials.best_trial))
file.write('\n')
file.write('\n')
file.close()

#trials.losses()
file = open("parameters/trials_losses.txt", "w")
file.write('Trials_losses')
file.write(str(trials.losses()))
file.write('\n')
file.write('\n')
file.close()
