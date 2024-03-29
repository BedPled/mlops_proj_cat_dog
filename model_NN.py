import os

from keras import optimizers
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential


def save_model(model):
    if not os.path.exists("Model/"):
        os.makedirs("Model/")
    model_json = model.to_json()
    with open("Model/model.json", "w") as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Model/weights.h5")
    print("Model and weights saved")
    return


def get_model(num_classes=2, learning_rate=1):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("sigmoid"))

    opt = optimizers.Adadelta(learning_rate=learning_rate, rho=0.95)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


if __name__ == "__main__":
    save_model(get_model())
