import numpy as np
import matplotlib.pyplot as plt

from models.regression.mars.mars import MARS


def main():

    # Generate some synthetic data
    np.random.seed(0)

    nobs = 250
    n_features = 2

    X = np.random.rand(nobs, n_features)
    y = (
        3 * X[:, 0]
        + np.sin(3 * X[:, 1])
        + np.cos(50 * X[:, 0] * X[:, 1])
        + 0.1 * np.random.randn(nobs)
    )

    # Fit MARS model
    mars = MARS(
        X,
        y,
        max_terms=4,  # 10,
        min_samples=3,
        max_interaction_term_degree=2,
        pruning=True,
        penalty=2.0,
    )
    mars.fit()

    # Make predictions
    y_hat = mars.predict(X)

    # Print the results
    print("Predictions:", y_hat)

    plt.plot(y_hat)
    plt.plot(y)
    plt.show()

    plt.plot(y_hat - y)
    plt.show()


# Example usage:
if __name__ == "__main__":
    main()
