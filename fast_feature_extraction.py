# -*- coding: utf-8 -*-

import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from hdf5_generator import HDF5Generator


def evaluate(path):
    model = load_model(path)
    with HDF5Generator('hdf5_data/test.hdf5', 34) as test_gen:
        eval_rs = model.evaluate(
            test_gen.generator(),
            steps=test_gen.db_size // test_gen.batch_size
        )
        print(eval_rs)


class FastFeatureExtractionGender(kt.HyperModel):
    def build(self, hp):
        inputs = keras.Input(shape=(4*4*512), name='input')
        first_dense = hp.Choice('first_dense', [4, 8, 16, 32, 64])
        dense = layers.Dense(first_dense, activation='relu')(inputs)
        dropout_rate = hp.Float('dropout_rate', min_value=.3, max_value=.8, step=.1)
        dense = layers.Dropout(dropout_rate)(dense)

        if first_dense in [32, 64]:
            with hp.conditional_scope('first_dense', [32, 64]):
                if hp.Boolean('extra_dense_layer'):
                    with hp.conditional_scope('conditional_scope', [True]):
                        dense = layers.Dense(hp.Choice('extra_dense', [4, 8, 16]), activation='relu')(dense)

        outputs = layers.Dense(1, activation='sigmoid', name='output')(dense)

        model = keras.Model(inputs=inputs, outputs = outputs)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model


def start_hyper_tuning():
    hp = FastFeatureExtractionGender()
    tuner = kt.BayesianOptimization(
        hp, objective='val_accuracy', max_trials=160, executions_per_trial=2, directory='hypermodel', overwrite=True)

    with HDF5Generator('hdf5_data/train.hdf5') as train_gen:
        with HDF5Generator('hdf5_data/valid.hdf5', 34) as valid_gen:
            tuner.search(
                train_gen.generator(),
                epochs=50,
                steps_per_epoch=train_gen.db_size // train_gen.batch_size,
                validation_data=valid_gen.generator(),
                validation_steps=valid_gen.db_size // valid_gen.batch_size,
                verbose=2,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath='gender_prediction_ffe_best.keras',
                        save_best_only='True',
                        monitor='val_accuracy'
                    )
                ])

    print(evaluate('gender_prediction_ffe_best.keras'))
