import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import requests


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


def evaluate_predictions(y_true, y_pred):
    """
    Evaluates the mean squared error between the values in y_true and the values
    in y_pred.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param y_true: Numpy array, the true target values from the test set;
    :param y_pred: Numpy array, the values predicted by your model.
    :return: float, the the mean squared error between the two arrays.
    """
    assert y_true.shape == y_pred.shape
    return ((y_true - y_pred) ** 2).mean()


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model

def download_from_onedrive(onedrive_link: str, destination: str):
    url = onedrive_link + "?download=1"
    headers = {
        "User-Agent": "Mozilla/5.0 "
                      "(Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 "
                      "(KHTML, like Gecko) "
                      "Chrome/70.0.3538.77 Safari/537.36"
    }
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    data_path = '../data/data.npz'
    x, y = load_data(data_path)

    ############################################################################
    # EDITABLE SECTION OF THE SCRIPT: if you need to edit the script, do it here
    ############################################################################

    ### Preparation of test data for prediction
    ## Load test data
    # Load test set x
    #with open('../src/test_set_x.npy', 'rb') as f:
    #    test_set_x = np.load(f)
    # Load test set y
    #with open('../src/test_set_y.npy', 'rb') as f:
    #    test_set_y = np.load(f)
    #x = test_set_x
    #y = test_set_y

    # Your model
    # Load the trained model
    baseline_model_path = './baseline_model.pickle'
    baseline_model = load_model(baseline_model_path)
    # Predict on the given samples
    y_pred = baseline_model.predict(x)
    # Evaluate the prediction using MSE
    mse = evaluate_predictions(y_pred, y)
    print('MSE of your model: {}'.format(mse))

    ##### LINEAR REGRESSION
    #### Load the trained model
    linear_regression_path = './linear_regression.pickle'
    linear_regression = load_model(linear_regression_path)

    # if linear_regression:
    
    ## Create 2 other vectors on test data
    test_x1_vector = x[:, 0]
    test_x2_vector = x[:, 1]

    # reshape test y
    test_set_y_reshape = y.reshape(-1, 1)

    # reshape test x1 vector
    test_x1_vector_reshape = test_x1_vector.reshape(-1, 1)
    #print(test_x1_vector_reshape.shape)

    # reshape test x2 vector
    test_x2_vector_reshape = test_x2_vector.reshape(-1, 1)
    #print(test_x2_vector_reshape.shape)

    # create test cos vector and reshape
    cos_vector_test = np.ones(test_x2_vector.shape)
    for i in range(len(cos_vector_test)):
        cos_vector_test[i] = np.cos(test_x2_vector[i])
    cos_vector_test_reshape = cos_vector_test.reshape(-1, 1)
    #print(cos_vector_test_reshape.shape)

    # create test x1x1 vector and reshape
    x1x1_vector_test = np.ones(test_x1_vector.shape)
    for i in range(len(x1x1_vector_test)):
        x1x1_vector_test[i] = test_x1_vector[i] * test_x1_vector[i]
    x1x1_vector_test_reshape = x1x1_vector_test.reshape(-1, 1)
    #print(x1x1_vector_test_reshape.shape)

    x_vectors_test = np.hstack((test_x1_vector_reshape, test_x2_vector_reshape, cos_vector_test_reshape, x1x1_vector_test_reshape))
    #print(x_vectors_test.shape)

    #### Predict on the given samples
    y_predicton_lr = linear_regression.predict(x_vectors_test)

    #else:

    ##### RANDOM FOREST REGRESSOR
    #### Load the trained model
    onedrive_url = "https://usi365-my.sharepoint.com/:u:/g/personal/hatast_usi_ch/EVcomMk8ohFKkWyRQLz519cByXdRvr-uuee02VASnunLqA"
    download_from_onedrive(onedrive_url, './random_forest_regressor.pickle')
    random_forest_regressor_path = './random_forest_regressor.pickle'
    random_forest_regressor = load_model(random_forest_regressor_path)

    ### PREDICT WITH TEST SET - correct solution -> MSE should be on test data
    y_predicton_rfr = random_forest_regressor.predict(x)

    # Evaluate the prediction for Random Forest Regressor using MSE
    mse = evaluate_predictions(y_predicton_rfr, y)
    print('MSE of Random Forest Regressor: {}'.format(mse))
    print("MSE below is for linear model")

    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################

    # Evaluate the prediction using MSE
    mse = evaluate_predictions(y_predicton_lr, y)
    print('MSE: {}'.format(mse))
