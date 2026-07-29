import datetime
import klax
import jax.random as jrandom
import time
import jax

import tmlsm.data as td
import tmlsm.plots as tp
import tmlsm.models as tm

now = datetime.datetime.now


def main():
    # Load and visalize data
    E_infty = 0.5
    E = 2.0
    eta = 1.0
    n = 100
    omegas = [1.0]
    As = [1.0]

    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    tp.plot_data(eps, eps_dot, sig, omegas, As)

    # Create a random key for the random weight initialization and the
    # batch generation.
    key = jrandom.PRNGKey(time.time_ns())
    keys = jrandom.split(key, 2)

    # Build model instance
    model = tm.build(key=keys[0])

    # Calibrate the model
    t1 = now()
    print(t1)

    model, history = klax.fit(
        model,
        ((eps, dts), sig),
        batch_axis=0,
        steps=1000,
        history=klax.HistoryCallback(log_every=1),
        key=keys[1],
    )

    t2 = now()
    print("it took", t2 - t1, "(sec) to calibrate the model")

    history.plot()

    # Unwrap all wrappers and apply all constraints to the model
    model_ = klax.finalize(model)

    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)

    As = [1, 1, 2]
    omegas = [1, 2, 3]

    eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)

    eps, eps_dot, sig, dts = td.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
    sig_m = jax.vmap(model_)((eps, dts))
    tp.plot_data(eps, eps_dot, sig, omegas, As)
    tp.plot_model_pred(eps, sig, sig_m, omegas, As)


if __name__ == "__main__":
    main()
