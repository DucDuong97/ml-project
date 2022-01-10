from task2.regression import RidgeRegression, transform
from task2.wine_dataset import vectorized_data
import numpy as np


def get_regression_model():
    # TODO: instantiate the regression model of your choice with the optimal set of parameters here and return it.
    C = 10
    return RidgeRegression(C)


def get_transformed_data():
    # TODO: this function should return your transformed version of the whole wine reviews dataset
    # You can either load the original dataset with the provided code and execute the transformations after that
    #
    # or you can transform your data once, save it somewhere and load it again in this method. Make sure
    # that you include it in your repository then or write some code that downloads it if it is not present.
    #
    # Please return data as two floating point numpy arrays X, y with X.shape==(n, d) and y.shape==(n,)
    data = vectorized_data()
    points = np.array(data.pop('points'))
    data = transform(data)
    return data, points
