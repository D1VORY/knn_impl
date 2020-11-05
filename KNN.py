import pandas as pd
import numpy as np
from typing import *
from statistics import mode, StatisticsError


def calculate_distance_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
      Computes distance between each row in x1 and x2
      the result is matrix, where [i][j] element = distance
      between x1[i] and x2[j]
      As a distance metric it uses Euclidian distance
    """
    polynomial_sum = np.sum(np.square(x1)[:, np.newaxis, :], axis=2) - 2 * x1.dot(x2.T) + np.sum(np.square(x2), axis=1)
    return np.sqrt(polynomial_sum)


def most_frequent_class(indeces: np.ndarray, labels: np.ndarray) -> float:
    """
      gets an array of indeces that represent the most closest
      objects and all train labels. Then uses that indeces to create a list with
      labels that correspond to them. Finally finds the most
      common label and returns it.
    """
    closest_labels = [labels[index, 0] for index in indeces]
    try:
        return mode(closest_labels)
    except StatisticsError:
        return closest_labels[0]


def knn_classifier(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, n: int = 7):
    """
      Assigns class labels to array of test objects based on k-nearest neighbors algorithm.

      :param x_train: np.ndarray, shape = (a, b)
      :param y_train: np.ndarray, shape = (a, 1)
      :param x_test: np.ndarray, shape = (c, b)
      :param y_test: np.ndarray, shape = (c, 1)
      :param n: int, number of neighbours
      :return:
    """
    # 1. find distance matrix
    # 2. get argsorted matrix of indexes
    # 3. get first n indexes of sorted matrix
    # 4. find majority class
    # 5. assign most frequent class to test element

    distance_matrix = calculate_distance_matrix(x_test, x_train)
    indeces_of_sorted = np.argsort(distance_matrix, axis=1)
    for i in range(y_test.shape[0]):
        y_test[i, 0] = most_frequent_class(indeces_of_sorted[i, :n], y_train)

    return y_test

if __name__ == '__main__':
  train_data = pd.read_csv('./susy100K.csv').to_numpy()
  test_data = pd.read_csv('./susytest1000.csv').to_numpy()

  x_train, y_train = train_data[:, 1:], train_data[:, :1]
  x_test, y_test = test_data, np.zeros(shape=(test_data.shape[0], 1))

  # print('train shape: ', x_train.shape)
  # print('train labels shape: ', y_train.shape)
  # print('test shape: ', x_test.shape)
  # print('test labels shape: ', y_test.shape)

  knn_classifier(x_train,y_train, x_test, y_test)


  pd.DataFrame(y_test).to_csv("labels.csv")
  print('Done')