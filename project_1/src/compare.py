from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold


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

    # split training set into 10 parts
    kf = KFold(n_splits=10, shuffle=True)
    fold_iterator = kf.split(x, y)

    # Utility to split the data
    acc_lr = []
    acc_rfr = []

    # iterating over the training set part by part
    for train_index, test_index in fold_iterator:

        # each iteration I split the training set into training and validation
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[test_index], y[test_index]

        ### TRAIN MODEL Random Forest Regressor WITH TRAINING SET
        rfr = RandomForestRegressor()
        rfr.fit(x_train, y_train)

        ## Preparing data for TRAINING MODEL Linear Regression WITH TRAINING SET
        x1_vector = x_train[:, 0]
        x2_vector = x_train[:, 1]

        # reshape y
        training_set_y_reshape = y_train.reshape(-1, 1)
        x1_vector_reshape = x1_vector.reshape(-1, 1)
        x2_vector_reshape = x2_vector.reshape(-1, 1)

        # create cos vector and reshape
        cos_vector = np.ones(x2_vector.shape)
        for i in range(len(cos_vector)):
            cos_vector[i] = np.cos(x2_vector[i])
        cos_vector_reshape = cos_vector.reshape(-1, 1)

        # create x1x1 vector and reshape
        x1x1_vector = np.ones(x1_vector.shape)
        for i in range(len(x1x1_vector)):
            x1x1_vector[i] = x1_vector[i] * x1_vector[i]
        x1x1_vector_reshape = x1x1_vector.reshape(-1, 1)

        x_vectors = np.hstack((x_train, cos_vector_reshape, x1x1_vector_reshape))
        #print(x_vectors.shape)

        ### TRAIN MODEL Linear Regression WITH TRAINING SET
        lr = LinearRegression(fit_intercept=True)
        lr.fit(x_vectors, y_train)

        ## Preparing data for VALIDATION MODEL Linear Regression WITH VALIDATION SET
        x1_vector_val = x_val[:, 0]
        x2_vector_val = x_val[:, 1]

        # reshape y
        validation_set_y_reshape = y_val.reshape(-1, 1)
        x1_vector_reshape = x1_vector_val.reshape(-1, 1)
        x2_vector_reshape = x2_vector_val.reshape(-1, 1)

        # create cos vector and reshape
        cos_vector_val = np.ones(x2_vector_val.shape)
        for i in range(len(cos_vector_val)):
            cos_vector_val[i] = np.cos(x2_vector_val[i])
        cos_vector_val_reshape = cos_vector_val.reshape(-1, 1)

        # create x1x1 vector and reshape
        x1x1_vector_val = np.ones(x1_vector_val.shape)
        for i in range(len(x1x1_vector_val)):
            x1x1_vector_val[i] = x1_vector_val[i] * x1_vector_val[i]
        x1x1_vector_val_reshape = x1x1_vector_val.reshape(-1, 1)

        x_vectors_val = np.hstack((x_val, cos_vector_val_reshape, x1x1_vector_val_reshape))
        #print(x_vectors.shape)

        # MEASURE ACCURACY
        current_acc_rfr = rfr.score(x_val, y_val) # accuracy_score
        acc_rfr.append(current_acc_rfr)
        current_acc_lr = lr.score(x_vectors_val, y_val)
        acc_lr.append(current_acc_lr)

    print("Linear Regression accuracy:       ", acc_lr)
    print("Random Forest Regressor accuracy: ", acc_rfr)

    print("Linear Regression average accuracy:       {:.3f} +- {:.3f}".format(np.mean(acc_lr), np.std(acc_lr)))
    print("Random Forest Regressor averace accuracy: {:.3f} +- {:.3f}".format(np.mean(acc_rfr), np.std(acc_rfr)))

    # Paired two sample test
    T, p_val = ttest_rel(acc_lr, acc_rfr)
    print('t-test: T={:.2f}, p-value={:.4f}'.format(T, p_val)) #tt
    print("is T={:.2f} in 95\% confidence interval (-1.96, 1.96) ?".format(T))
    # Null hypothesis
    # I define significant level, e. g. 0.05
    # If p-value is lower than 0.05, null hypothesis is rejected based on available data

    # T is -38.45 which is outside the 95% confidence interval (-1.96, 1.96) ->
    # we reject the null hypothesis because we can not say which model is statistically better
    # and hence select model with smaller variance = MSE