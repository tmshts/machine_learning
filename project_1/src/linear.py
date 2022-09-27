import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib as mpl


def the_model(x):
    y = 1.764 + (-0.164) * x[:, 0] + (-0.652) * x[:, 1] + (-0.019) * np.cos(x[:, 1]) + (0.042) * x[:, 0] * x[:, 0]
    return y

def save_sklearn_model(model, filename):
    """
    Saves a Scikit-learn model to disk.
    Example of usage:
    
    >>> reg = sklearn.linear_models.LinearRegression()
    >>> reg.fit(x_train, y_train)
    >>> save_sklearn_model(reg, 'my_model.pickle')    

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    joblib.dump(model, filename)

if __name__ == '__main__':
    # Load training set x
    with open('../src/training_set_x.npy', 'rb') as f:
        training_set_x = np.load(f)

    # Load training set y
    with open('../src/training_set_y.npy', 'rb') as f:
        training_set_y = np.load(f)

    # Load test set x
    with open('../src/test_set_x.npy', 'rb') as f:
        test_set_x = np.load(f)

    # Load test set y
    with open('../src/test_set_y.npy', 'rb') as f:
        test_set_y = np.load(f)

    # to visualize the data in order to check what I can expect
    x1_vector = training_set_x[:, 0]
    x2_vector = training_set_x[:, 1]
    #plt.scatter(x1_vector, training_set_y, c='b', marker='.')
    #plt.scatter(x2_vector, training_set_y, c='r', marker='*')
    #plt.xlabel("x values")
    #plt.ylabel("y values")
    #plt.show()

    # reshape y
    #print(training_set_y.shape)
    training_set_y_reshape = training_set_y.reshape(-1, 1)
    #print(training_set_y_reshape.shape)

    # reshape x1 vector
    #print(x1_vector.shape)
    x1_vector_reshape = x1_vector.reshape(-1, 1)
    #print(x1_vector_reshape.shape)

    # reshape x2 vector
    #print(x2_vector.shape)
    x2_vector_reshape = x2_vector.reshape(-1, 1)
    #print(x2_vector_reshape.shape)

    # create cos vector and reshape
    cos_vector = np.ones(x2_vector.shape)
    for i in range(len(cos_vector)):
        cos_vector[i] = np.cos(x2_vector[i])
    cos_vector_reshape = cos_vector.reshape(-1, 1)
    #print(cos_vector_reshape.shape)

   # create x1x1 vector and reshape
    x1x1_vector = np.ones(x1_vector.shape)
    for i in range(len(x1x1_vector)):
        x1x1_vector[i] = x1_vector[i] * x1_vector[i]
    x1x1_vector_reshape = x1x1_vector.reshape(-1, 1)
    #print(x1x1_vector_reshape.shape)

    # pack all the vectors into 4-D matrix
    x_vectors = np.hstack((training_set_x, cos_vector_reshape, x1x1_vector_reshape))
    #print(x_vectors.shape)

    ### TRAIN MODEL WITH TRAINING SET - I get thetas
    lr = LinearRegression(fit_intercept=True)
    lr.fit(x_vectors, training_set_y)
    thetas = [lr.intercept_, lr.coef_]
    print('thetas = {}'.format(thetas))

    # Save the linear regression model
    save_sklearn_model(lr, '../deliverable/linear_regression.pickle')

    ### PREDICT WITH TRAINING SET - just for my comparisson
    # estimated response
    #y_est = lr.predict(x_vectors)

    # calculate mean square error as performance function
    #mse_training = ((y_est - training_set_y)**2).mean()
    #print("Mean square error: ", mse_training)

    # different option
    #mse = mean_squared_error(training_set_y, y_est)
    #print(mse)


    ### PREDICT WITH TEST SET - correct solution
    test_x1_vector = test_set_x[:, 0]
    test_x2_vector = test_set_x[:, 1]

    # reshape test y
    #print(test_set_y.shape)
    test_set_y_reshape = test_set_y.reshape(-1, 1)
    #print(test_set_y_reshape.shape)

    # reshape test x1 vector
    #print(test_x1_vector.shape)
    test_x1_vector_reshape = test_x1_vector.reshape(-1, 1)
    #print(test_x1_vector_reshape.shape)

    # reshape test x2 vector
    #print(test_x2_vector.shape)
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

    # pack all the vectors into 4-D matrix
    x_vectors_test = np.hstack((test_set_x, cos_vector_test_reshape, x1x1_vector_test_reshape))
    #print(x_vectors_test.shape)

    ### PREDICT linear function
    y_predict = lr.predict(x_vectors_test)
    #print(y_est)

    mse = mean_squared_error(test_set_y, y_predict)
    print(mse)


    # 2-d for x1_vector and x2_vector
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=5., azim=0)
    ax.set_xlabel('x1_vector')
    ax.set_ylabel('x2_vector')
    ax.set_zlabel('y_predict')
    ax.plot_trisurf(test_set_x[:, 0], test_set_x[:, 1], y_predict, alpha=0.3, cmap=mpl.colormaps['viridis'], label='predict fun')
    ax.scatter(test_set_x[:, 0], test_set_x[:, 1], test_set_y, c='k', marker='.')
    plt.show()