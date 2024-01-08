import mlflow
from get_dataset import get_dataset
from model_NN import get_model, save_model


def train_model(model, x_train, x_test, y_train, y_test):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    mlflow.set_experiment("MLflow EXP")
    with mlflow.start_run():
        from keras.preprocessing.image import ImageDataGenerator

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
            generated_data.flow(x_train, y_train, batch_size=8),
            steps_per_epoch=x_train.shape[0] // 8,
            epochs=25,
            validation_data=(x_test, y_test),
        )

        return model


def main():
    x_train, x_test, y_train, y_test = get_dataset()
    model = get_model(len(y_train[0]))
    model = train_model(model, x_train, x_test, y_train, y_test)
    save_model(model)
    return model


if __name__ == "__main__":
    main()
