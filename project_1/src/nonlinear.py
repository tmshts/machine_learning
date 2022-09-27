import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib as mpl

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

    # Let's look at the data
    #plt.scatter(training_set_x[:, 0], training_set_x[:, 1], c=training_set_y, edgecolors='k')
    #plt.xlabel(r'$x_1$')
    #plt.ylabel(r'$x_2$')
    #plt.show()

    rfr = RandomForestRegressor(n_estimators=650) # default MSE as criterion

    ### TRAIN MODEL WITH TRAINING SET
    rfr.fit(training_set_x, training_set_y)

    # Save the random forest regressor model
    save_sklearn_model(rfr, '../deliverable/random_forest_regressor.pickle')

    ## PREDICT WITH TRAINING SET - just to compare -> it is better MSE because we used the same data for training
    #y_predict = rfr.predict(training_set_x)
    #mse = mean_squared_error(training_set_y, y_predict)
    #print(mse)

    ### PREDICT WITH TEST SET - correct solution -> MSE should we get from test data
    y_predict = rfr.predict(test_set_x)
    #print(y_predict)

    mse = mean_squared_error(test_set_y, y_predict)
    print(mse)


    #import pandas as pd
    #feature_list = list(training_set_x.columns)
    #feature_imp = pd.Series(rfr.feature_importances_, index=feature_list).sort_values
    #print(feature_imp)

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
