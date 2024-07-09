from abc import ABC
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ...models import RegressionModel


class RandomForestTreeRegressor(RegressionModel, ABC):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:

        # Number of trees in the forest
        self.n_estimators = n_estimators

        # Maximum depth of the trees
        self.max_depth = max_depth

        # Random state for reproducibility
        self.random_state = random_state

        # Placeholder for the RandomForestRegressor model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        return

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegressionModel:
        """
        Fit the Random Forest Regressor model to the data X, y.
        """

        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output for input data X.
        """
        return self.model.predict(X)
