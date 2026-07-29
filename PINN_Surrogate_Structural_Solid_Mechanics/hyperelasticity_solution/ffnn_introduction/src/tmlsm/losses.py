"""Implementations of objective functions."""

import jax
import jax.numpy as jnp
import klax


class MSE(klax.Loss):
    """Reimplementation of `klax.MSE` loss function."""

    def __call__(self, model, batch, batch_axis):
        x, y = batch
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))
