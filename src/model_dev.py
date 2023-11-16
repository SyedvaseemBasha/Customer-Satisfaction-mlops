import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models"""

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            x_train: Training data
            y_train: Training data
        Returns:
              None     
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            x_train: Training data
            y_train: Training data
        Returns:
              None     
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training completed")
            return reg
        except Exception as e:
            logging.error("error in Training model:{}".format(e))
            raise e
