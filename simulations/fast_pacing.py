"""
Fast pacing using the Numba-JIT ORd model.

Uses scipy Radau with the JIT-compiled RHS, giving ~50-100x speedup
over the pure-Python version for block sweep simulations.
"""

import numpy as np
from scipy.integrate import solve_ivp

from models.ord_model_numba import rhs_numba, params_to_array, _bt, _Acap, _F, _vjsr
from models.ord_model import initial_conditions

STIM_DUR = 0.5    # ms
STIM_AMP = -80.0  # pA/pF


def pace_beat_fast(y0, cl, p_arr, ikr_block=0.0, dense=True):
    """
    Integrate one beat using numba-compiled RHS.
    p_arr: numpy array from params_to_array(param_dict).
    """
    t_stim_on  = 5.0
    t_stim_off = 5.0 + STIM_DUR

    def _ode(Istim_val):
        def ode(t, y):
            return rhs_numba(t, y, p_arr, Istim_val, ikr_block)
        return ode

    def _run(t0, t1, y_init, Istim_val, t_eval_seg):
        sol = solve_ivp(
            _ode(Istim_val), [t0, t1], y_init,
            method="Radau",
            t_eval=t_eval_seg if len(t_eval_seg) > 0 else None,
            rtol=1e-6, atol=1e-8,
            max_step=np.inf,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.t, sol.y

    if dense:
        n_pts = max(int(cl / 0.2) + 1, 200)
        t_full = np.linspace(0.0, cl, n_pts)
    else:
        t_full = np.linspace(0.0, cl, 20)

    def _seg_eval(t0, t1):
        return t_full[(t_full >= t0) & (t_full <= t1)]

    t_s1 = _seg_eval(0.0, t_stim_on)
    t1_arr, y1_arr = _run(0.0, t_stim_on, y0, 0.0, t_s1)

    t_s2 = _seg_eval(t_stim_on, t_stim_off)
    t2_arr, y2_arr = _run(t_stim_on, t_stim_off, y1_arr[:, -1], STIM_AMP, t_s2)

    t_s3 = _seg_eval(t_stim_off, cl)
    t3_arr, y3_arr = _run(t_stim_off, cl, y2_arr[:, -1], 0.0, t_s3)

    t_out = np.concatenate([t1_arr, t2_arr[1:], t3_arr[1:]])
    y_out = np.concatenate([y1_arr, y2_arr[:, 1:], y3_arr[:, 1:]], axis=1)
    return t_out, y_out


def run_ss_fast(param_dict, cl=1000.0, n_beats=80, n_save=1, ikr_block=0.0, y0=None):
    """
    Run to steady state using the fast numba model.
    API matches simulations.pacing.run_steady_state.
    """
    p_arr = params_to_array(param_dict)

    if y0 is None:
        y0 = initial_conditions()

    y_cur = y0.copy()
    t_all = []
    y_all = []

    for beat_idx in range(n_beats):
        is_last = beat_idx >= n_beats - n_save
        t_b, y_b = pace_beat_fast(y_cur, cl, p_arr, ikr_block=ikr_block, dense=is_last)
        y_cur = y_b[:, -1]

        if is_last:
            offset = (beat_idx - (n_beats - n_save)) * cl
            t_all.append(t_b + offset)
            y_all.append(y_b)

    return np.concatenate(t_all), np.concatenate(y_all, axis=1)
