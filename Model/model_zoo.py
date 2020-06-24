import os
from tensorflow.keras import optimizers, initializers, regularizers, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelZoo:
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def select_model(self, model_name):
        if model_name == "alex_net":
            print("model alex net was selected")
            self.build_alex_net()
        elif model_name == "vgg_net":
            print("model vgg net was selected")
            self.build_vgg_net()
        else:
            print("select different model!")

    def build_alex_net(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=96, input_shape=(self.image_shape[0], self.image_shape[1], 3), kernel_size=(11, 11), strides=4, padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        self.model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        self.model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        self.model.add(Flatten())

        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Dense(1000))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

        print(self.model.summary())

        self.compile()

    def build_vgg_net(self):
        self.model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01), input_shape=(self.image_shape[0], self.image_shape[1], 3)),
            Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(2, 2),

            Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(2, 2),

            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(2, 2),

            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(4096, kernel_initializer="he_normal"),
            Dense(2048, kernel_initializer="he_normal"),
            Dense(1024, kernel_initializer="he_normal"),
            Dense(3, activation='softmax')
        ])

        print(self.model.summary())
        self.compile()

    def compile(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy']
                           )

        self.define_callbacks()

    def define_callbacks(self):
        early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', baseline=0.1, patience=30)

        checkpoint_path = os.path.join("training_checkpoints")
        checkpoint_perfix = os.path.join(checkpoint_path, "ckpt_{epoch}")
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_perfix, save_weights_only=True, save_best_only=True
        )

        self.my_callbakcs = [
            early_stopping_callback,
            checkpoint_callback
        ]





