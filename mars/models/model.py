from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class RegressionModel(Model, ABC):
    pass


class ClassificationModel(Model, ABC):
    pass


class RankModel(Model, ABC):
    pass
