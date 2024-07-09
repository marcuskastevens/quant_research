from abc import ABC
from typing import List

import numpy as np


class BasisFunction(ABC):

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HingeBasisFunction(BasisFunction):

    def __init__(self, variable: int, knot: float, direction: str):

        # Variable index
        self.variable = variable

        # Split value (knot)
        self.knot = knot

        # Split direction ('+' or '-')
        self.direction = direction

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the basis function on input data X.
        """

        if self.direction == "+":
            # Evaluate the positive hinge function
            return np.maximum(0, X[:, self.variable] - self.knot)
        else:
            # Evaluate the negative hinge function
            return np.maximum(0, self.knot - X[:, self.variable])


class BasisInteractionFunction(BasisFunction):

    def __init__(self, basis_functions: List[BasisFunction]) -> None:

        self.basis_functions = basis_functions

        return

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the interaction basis function on input data X."""

        interaction_term = np.ones(X.shape[0])

        # Multiply the evaluated bases together
        for basis_function in self.basis_functions:
            interaction_term *= basis_function.evaluate(X=X)

        return interaction_term
