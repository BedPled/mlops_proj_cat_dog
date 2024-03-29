import numpy as np
import pandas as pd
from keras.models import model_from_json

from get_dataset import get_img


def predict(model, obj):
    pred = model.predict(obj)
    pred = np.argmax(pred, axis=1)
    pred = "cat" if pred[0] == 0 else "dog"
    return pred


if __name__ == "__main__":
    img_dir = "./dataset/test_cat.jpg"

    img = get_img(img_dir)

    X_data = np.zeros((1, 64, 64, 3), dtype="float64")
    X_data[0] = img

    # Get model
    model_file = open("Model/model.json", "r")
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    model.load_weights("../Model/weights.h5")

    pred_category = predict(model, X_data)
    pd.DataFrame({"out": [pred_category]}).to_csv("out.csv")
    # print("It is a " + pred_category + "!")
