# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A minimalist example with sklearn to show how to develop and explore a model with exca.
"""
import typing as tp
import numpy as np
import pydantic
import sys
import exca
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


class Dataset(pydantic.BaseModel):
    n_samples: int = 100
    noise: float = 0.1
    random_state: int = 42
    test_size: float = 0.2
    model_config = pydantic.ConfigDict(extra="forbid")

    def get(self) -> tp.Tuple[np.ndarray]:
        # Generate synthetic data
        X, y = make_regression(
            n_samples=self.n_samples,
            noise=self.noise,
            random_state=self.random_state
        )
        # Split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test


class Model(pydantic.BaseModel):
    data: Dataset = Dataset()
    alpha: float = 1.0
    max_iter: int = 1000
    infra: exca.TaskInfra = exca.TaskInfra(folder='.cache/')

    @infra.apply
    def score(self):
        # Get data
        X_train, X_test, y_train, y_test = self.data.get()
        
        # Train a Ridge regression model
        print('Fit...')
        model = Ridge(alpha=self.alpha, max_iter=self.max_iter)
        model.fit(X_train, y_train)

        # Evaluate
        print('Score...')
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse


def args_to_nested_dict(args: list[str]) -> tp.Dict[str, tp.Any]:
    """
    Parses a list of Bash-style arguments (e.g., --key=value) into a nested dict.
    """
    nested_dict = {}
    for arg in args:
        # Split argument into key and value
        key, value = arg.lstrip("--").split("=", 1)
        # Convert flat key into a nested dictionary
        keys = key.split(".")
        current_level = nested_dict
        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})
        current_level[keys[-1]] = value
    return nested_dict


if __name__ == "__main__":
    # Validate config
    config = exca.ConfDict.from_args(sys.argv[1:])
    model = Model(**config)
    print(model.infra.config)

    # Score
    mse = model.score()
    print(mse)
