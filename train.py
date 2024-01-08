import os
# from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import ModelCheckpoint

from model_NN import get_model, save_model
from get_dataset import get_dataset

import mlflow
import numpy

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def train_model(model, x_train, x_test, y_train, y_test):
    mlflow.set_experiment("MLflow EXP")
    with mlflow.start_run():
        # checkpoints = []
        # if not os.path.exists('Data/Checkpoints/'):
        #     os.makedirs('Data/Checkpoints/')
        # checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
        # checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

        # Creates live data:
        # For better yield. The duration of the training is extended.

        # If you don't want, use this:
        # model.fit(x_train, y_train,
        #           batch_size=8,
        #           steps_per_epoch=x_train.shape[0] // 8,
        #           epochs=25,
        #           validation_data=(x_test, y_test),
        #           # shuffle=True,
        #           # callbacks=checkpoints
        #           )

        from keras.preprocessing.image import ImageDataGenerator
        generated_data = ImageDataGenerator(featurewise_center=False,
                                            samplewise_center=False,
                                            featurewise_std_normalization=False,
                                            samplewise_std_normalization=False,
                                            zca_whitening=False,
                                            rotation_range=0,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            horizontal_flip=True,
                                            vertical_flip=False)
        generated_data.fit(x_train)

        model.fit_generator(generated_data.flow(x_train, y_train, batch_size=8),
                            steps_per_epoch=x_train.shape[0]//8,
                            epochs=25,
                            validation_data=(x_test, y_test),
                            # callbacks=checkpoints
                            )

        return model

def main():
    x_train, x_test, y_train, y_test = get_dataset()
    model = get_model(len(y_train[0]))
    model = train_model(model, x_train, x_test, y_train, y_test)
    save_model(model)
    return model


if __name__ == '__main__':
    main()





# --пример как должно быть---------------------------------------------------
# import os

# import mlflow
# import pandas as pd
# from hydra import compose, initialize
# from mlflow.models import infer_signature
#
# import model_NN

# import feature_generation
# import metrics
# import target_generation


# def main():
#     """
#     Функция реализует обучения модели & логгирование метрик.
#     """
#     initialize(version_base=None, config_path="../configs")
#     cfg = compose(config_name="config.yaml")
#     cfg_mlflow = compose(config_name="mlflow.yaml")
#     cfg_catboost = compose(config_name="catboost_params.yaml")
#
#     drop_features = cfg["modeling"]["drop_columns"]
#     target = cfg["modeling"]["target"]
#
#     data = pd.read_parquet(cfg["paths"]["sells"])
#
#     # generate features
#     features = feature_generation.apply_feature_generation(
#         data=data,
#         target=cfg["constants"]["raw_target"],
#         predicting_unit=cfg["constants"]["predicting_unit"],
#         date_col=cfg["constants"]["date_col"],
#         rolling_windows=cfg["feature_generation"]["rolling_windows"],
#     )
#
#     # create target
#     features = target_generation.create_target(
#         data=features,
#         horizont=cfg["constants"]["horizont"],
#         raw_target=cfg["constants"]["raw_target"],
#         predicting_unit=cfg["constants"]["predicting_unit"],
#     )
#
#     features_list = list(
#         features.loc[:, ~features.columns.isin([target] + drop_features)].columns
#     )
#
#     train_data = features[~features[target].isna()][features_list]
#     train_target = features[~features[target].isna()][target]
#
#     model = CatBoostRegressor(**cfg_catboost["catboost_params"])
#
#     model.fit(train_data, train_target)
#
#     model.save_model(os.path.join(cfg["paths"]["models"], "catboost.cbm"), format="cbm")
#
#     if cfg_mlflow["mlflow"]["logging"]:
#         # set tracking server uri for logging
#         mlflow.set_tracking_uri(uri=cfg_mlflow["mlflow"]["logging_uri"])
#
#         # create a new MLflow Experiment
#         mlflow.set_experiment(cfg_mlflow["mlflow"]["experiment_name"])
#
#         # start an MLflow run
#         with mlflow.start_run():
#             # log the hyperparameters
#             mlflow.log_params(cfg_catboost["catboost_params"])
#
#             # calculate metrics
#             all_metrics = metrics.Metrics(
#                 actual=train_target, prediction=model.predict(train_data)
#             )
#
#             # Log the loss metric
#             mlflow.log_metric("WAPE", all_metrics.wape())
#             mlflow.log_metric("MedianApe", all_metrics.median_ape())
#             mlflow.log_metric("MAE", all_metrics.mae())
#
#             # set a tag
#             mlflow.set_tag(
#                 cfg_mlflow["mlflow"]["tag_name"], cfg_mlflow["mlflow"]["tag_value"]
#             )
#
#             # Infer the model signature
#             signature = infer_signature(train_data, model.predict(train_data))
#
#             # Log the model
#             mlflow.sklearn.log_model(
#                 sk_model=model,
#                 artifact_path=cfg_mlflow["mlflow"]["artifact_path"],
#                 signature=signature,
#                 input_example=train_data,
#                 registered_model_name=cfg_mlflow["mlflow"]["registered_model_name"],
#             )
#
#
# if __name__ == "__main__":
#     main()
