import os

import hydra
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from omegaconf import DictConfig

from get_dataset import get_dataset
from model_NN import get_model, save_model


def train_model(model, x_train, x_test, y_train, y_test, epochs=25, batch_size=8):
    checkpoints = []
    if not os.path.exists("checkpoints/"):
        os.makedirs("checkpoints/")
    checkpoints.append(
        ModelCheckpoint(
            "checkpoints/best_weights.h5",
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            period=1,
        )
    )

    checkpoints.append(
        TensorBoard(
            log_dir="checkpoints/./logs",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
        )
    )

    generated_data = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
    )

    generated_data.fit(x_train)
    model.fit_generator(
        generated_data.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=checkpoints,
        shuffle=True,
    )

    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    x_train, x_test, y_train, y_test = get_dataset(test_size=cfg.dataset.test_size)
    model = get_model(num_classes=cfg.model.num_classes, learning_rate=cfg.model.learning_rate)

    model = train_model(model, x_train, x_test, y_train, y_test, epochs=cfg.model.epochs, batch_size=cfg.model.batch_size)
    save_model(model)


if __name__ == "__main__":
    main()
