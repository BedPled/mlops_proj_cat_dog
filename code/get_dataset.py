import numpy as np
from os import listdir
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def get_img(data_path: str):

    # Getting image array from path:
    img_size = 64
    img = io.imread(data_path)
    img = resize(img, (img_size, img_size, 3))
    return img


def get_dataset(dataset_path: str = 'dataset'):

    # Getting all data from data path:
    try:
        x_data = np.load('../np_dataset/X_data.npy')
        y_data = np.load('../np_dataset/Y_data.npy')

    except OSError:
        labels = listdir(dataset_path)  # Geting labels
        print('Categories:\n', labels)
        len_data = 0

        for label in labels:
            len_data += len(listdir(dataset_path + '/' + label))

        x_data = np.zeros((len_data, 64, 64, 3), dtype='float64')
        y_data = np.zeros(len_data)
        count_data = 0
        count_categories = [-1, '']  # For encode labels

        for label in labels:
            data_path = dataset_path + '/' + label

            for data in listdir(data_path):
                img = get_img(data_path + '/' + data)
                x_data[count_data] = img

                # For encode labels:
                if label != count_categories[1]:
                    count_categories[0] += 1
                    count_categories[1] = label
                y_data[count_data] = count_categories[0]
                count_data += 1

        # Create dateset:
        import keras
        import os

        y_data = keras.utils.to_categorical(y_data)

        if not os.path.exists('../np_dataset/'):
            os.makedirs('../np_dataset/')

        np.save('../np_dataset/X_data.npy', x_data)
        np.save('../np_dataset/Y_data.npy', y_data)

    x_data /= 255.
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    return x_train, x_test, y_train, y_test


def main():
    get_dataset()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
