"""Model implementations."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax


class Cell(eqx.Module):
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        self.layers = (
            klax.nn.Linear(3, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 2, weight_init=he_normal(), key=key),
        )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, gamma, x):
        eps = x[0]
        h = x[1]

        x = jnp.array([gamma, eps, h])

        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        print("This is the output: ", x)

        gamma = x[0]
        sig = x[1]

        return gamma, sig


class Model(eqx.Module):
    cell: Callable

    def __init__(self, *, key: PRNGKeyArray):
        self.cell = Cell(key=key)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        return ys


def build(*, key: PRNGKeyArray):
    """Make and return a model instance."""
    return Model(key=key)
