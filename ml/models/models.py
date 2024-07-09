"""
Module implementing abstract model interfaces.
"""

# Builtin dependencies
from abc import ABC, abstractmethod

# External dependencies
import numpy as np


class Model(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ClassificationModel(Model, ABC):
    pass


class ClusteringModel(Model, ABC):
    pass


class DimensionalityReductionModel(Model, ABC):
    pass


class RankingModel(Model, ABC):
    pass


class RegressionModel(Model, ABC):
    pass
