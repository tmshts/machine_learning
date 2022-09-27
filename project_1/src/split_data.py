import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y

if __name__ == '__main__':
    data_path = '../data/data.npz'
    x, y = load_data(data_path)

    #print(x.shape)
    #print(y.shape)

    # Split data set by using test_train_split() into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

    # Save training set x
    with open("../src/training_set_x.npy", "wb") as f:
        np.save(f, x_train)

    # Save training set y
    with open("../src/training_set_y.npy", "wb") as f:
        np.save(f, y_train)

    # Save test set x
    with open("../src/test_set_x.npy", "wb") as f:
        np.save(f, x_test)

    # Save test set y
    with open("../src/test_set_y.npy", "wb") as f:
        np.save(f, y_test)    