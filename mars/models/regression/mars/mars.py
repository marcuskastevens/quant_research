import itertools
from abc import ABC
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from ...models import RegressionModel
from .basis_functions import BasisFunction, HingeBasisFunction, BasisInteractionFunction


class MARS(RegressionModel):
    def __init__(
        self,
        max_terms: Optional[int] = 5,
        max_interaction_term_degree: Optional[int] = 1,
        pruning: Optional[bool] = True,
        penalty: Optional[float] = 2.0,
    ) -> None:

        # Maximum number of terms in the MARS model
        self.max_terms = max_terms

        # Maximum degree of interaction
        self.max_interaction_term_degree = max_interaction_term_degree

        # Whether to use pruning
        self.pruning = pruning

        # Penalty parameter for GCV
        self.penalty = penalty

        # List to store basis functions
        self.terms: List[BasisFunction] = []

        # Placeholder for the features and response variable
        self.X: np.ndarray = None
        self.y: np.ndarray = None

        # Define shape attribites
        self.n_samples: int = None
        self.n_features: int = None

        # Placeholder for the linear model
        self.model = None

        return

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegressionModel:
        """
        Fit the MARS model to the data X, y.
        """

        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape

        # Start with constant term
        self.terms = [np.ones(shape=self.n_samples)]

        # Construct the initial design matrix
        H = self._design_matrix(self.X)

        # Fit the initial linear model
        self.model = LinearRegression().fit(X=H, y=self.y)

        for _ in range(self.max_terms - 1):

            best_basis = None
            best_model = None
            best_error = np.inf

            # Generate candidate basis functions
            candidate_bases = self._generate_candidate_bases(self.n_features)

            for basis in candidate_bases:

                # Add the candidate basis to the design matrix
                H_new = self._add_basis(H, basis, self.X)

                # Fit the linear model with the new basis
                model = LinearRegression().fit(H_new, self.y)

                # Calculate the prediction error
                error = np.mean((self.y - model.predict(H_new)) ** 2)

                if error < best_error:

                    # Update the best basis function and model
                    best_error = error
                    best_basis = basis
                    best_model = model

            if best_basis is not None:

                # Add the best basis function to the terms
                self.terms.append(best_basis)

                # Update the design matrix with the best basis function
                H = self._add_basis(H, best_basis, self.X)

                # Update the model with the best model
                self.model = best_model

        if self.pruning:

            # Perform backward pass pruning if enabled
            self._backward_pass()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output for input data X.
        """

        # Construct the design matrix from the current terms (i.e., build design matrix from basis hinge function evalutions)
        H = self._design_matrix(X)

        # Return the predicted values
        return self.model.predict(H)

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construct the design matrix from the current terms.
        """

        # Evaluate each term in the current terms list
        return np.column_stack(
            [
                term.evaluate(X) if isinstance(term, BasisFunction) else term
                for term in self.terms
            ]
        )

    def _add_basis(
        self, H: np.ndarray, basis: BasisFunction, X: np.ndarray
    ) -> np.ndarray:
        """Add a new basis function to the design matrix."""

        # Add the evaluated basis function to the design matrix
        return np.column_stack([H, basis.evaluate(X)])

    def _generate_candidate_bases(self, n_features: int) -> List[BasisFunction]:
        """Generate candidate basis functions including interactions."""

        candidate_bases = []

        # Generate candidate basis functions for each variable and knot
        for variable in range(n_features):
            for knot in np.unique(self.X[:, variable]):
                for direction in ["+", "-"]:
                    candidate_bases.append(
                        HingeBasisFunction(variable, knot, direction)
                    )

        # Generate interaction basis functions if interaction degree is greater than 1
        if self.max_interaction_term_degree > 1:
            interaction_bases = self._generate_interaction_bases(candidate_bases)
            candidate_bases += interaction_bases

        return candidate_bases

    def _generate_interaction_bases(
        self, candidate_bases: List[BasisFunction]
    ) -> List[BasisFunction]:
        """Generate interaction basis functions up to the specified degree."""

        interaction_basis_functions = []

        # Generate interaction bases for each degree
        for degree in range(2, self.max_interaction_term_degree + 1):
            for basis_function_combination in itertools.combinations(
                candidate_bases, degree
            ):
                # Combine multiple basis functions into an interaction term
                interaction_basis_functions.append(
                    BasisInteractionFunction(basis_functions=basis_function_combination)
                )

        return interaction_basis_functions

    def _backward_pass(self):
        """Prune the model using backward pass with Generalized Cross-Validation (GCV)."""
        # Construct the design matrix from the current terms
        H = self._design_matrix(self.X)

        # Initialize the best terms and model
        best_terms = list(self.terms)
        best_model = self.model
        min_gcv = self._calculate_gcv(H, self.y)

        # Iterate over the terms to prune one by one
        for i in range(1, len(self.terms)):  # Skip the constant term
            terms_temp = [term for j, term in enumerate(self.terms) if j != i]
            H_temp = np.column_stack(
                [
                    term.evaluate(self.X) if isinstance(term, BasisFunction) else term
                    for term in terms_temp
                ]
            )
            model_temp = LinearRegression().fit(H_temp, self.y)
            gcv_temp = self._calculate_gcv(H_temp, self.y, model_temp)

            if gcv_temp < min_gcv:
                # Update the best terms and model if GCV is improved
                min_gcv = gcv_temp
                best_terms = terms_temp
                best_model = model_temp

        # Set the final pruned terms and model
        self.terms = best_terms
        self.model = best_model

    def _calculate_gcv(
        self, H: np.ndarray, y: np.ndarray, model: LinearRegression = None
    ) -> float:
        """Calculate the Generalized Cross-Validation (GCV) score."""
        if model is None:
            model = self.model

        n_samples = len(y)

        # Calculate the residual sum of squares
        rss = np.sum((y - model.predict(H)) ** 2)

        # Number of parameters in the model
        n_params = H.shape[1]

        # Effective number of parameters with penalty
        effective_params = n_params + self.penalty * (n_params - 1) / 2

        # Calculate the GCV score
        gcv = rss / (n_samples * (1 - (effective_params / n_samples)) ** 2)

        return gcv
