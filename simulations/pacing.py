"""
Pacing protocol and analysis utilities for the ORd model.

Uses a beat-by-beat integration approach with Radau (stiffly stable) solver,
which handles the multi-timescale dynamics of cardiac AP models robustly.
"""

import numpy as np
from scipy.integrate import solve_ivp
from models.ord_model import rhs, initial_conditions

STIM_DUR = 0.5    # ms
STIM_AMP = -80.0  # pA/pF


def pace_one_beat(y0, cl, params, ikr_block=0.0, dense=True):
    """
    Integrate one pacing beat starting from y0.

    Splits at stimulus boundaries so Radau never skips the 0.5 ms stimulus
    window regardless of step size.

    Parameters
    ----------
    dense : bool — if True return fine t_eval (for analysis); if False,
                   return only endpoint (faster burn-in)

    Returns (t_array, y_array) with t relative to beat start.
    """
    t_stim_on  = 5.0
    t_stim_off = 5.0 + STIM_DUR

    def _ode(Istim_val):
        def ode(t, y):
            return rhs(t, y, params, Istim=Istim_val, ikr_block=ikr_block)
        return ode

    def _run(t0, t1, y_init, Istim_val, t_eval_seg):
        # max_step: stimulus segment uses tight cap (0.5 ms window);
        # pre/post segments allow large adaptive steps — Radau's error
        # control still forces small steps during the fast upstroke.
        seg_dur = t1 - t0
        ms = 0.4 if seg_dur <= 1.0 else np.inf
        sol = solve_ivp(
            _ode(Istim_val), [t0, t1], y_init,
            method="Radau",
            t_eval=t_eval_seg if len(t_eval_seg) > 0 else None,
            rtol=1e-6, atol=1e-8,
            max_step=ms,
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
        mask = (t_full >= t0) & (t_full <= t1)
        return t_full[mask]

    # Segment 1: pre-stimulus [0, t_stim_on]
    t_s1 = _seg_eval(0.0, t_stim_on)
    t1_arr, y1_arr = _run(0.0, t_stim_on, y0, 0.0, t_s1)

    # Segment 2: stimulus [t_stim_on, t_stim_off]
    t_s2 = _seg_eval(t_stim_on, t_stim_off)
    t2_arr, y2_arr = _run(t_stim_on, t_stim_off, y1_arr[:, -1], STIM_AMP, t_s2)

    # Segment 3: post-stimulus [t_stim_off, cl]
    t_s3 = _seg_eval(t_stim_off, cl)
    t3_arr, y3_arr = _run(t_stim_off, cl, y2_arr[:, -1], 0.0, t_s3)

    # Concatenate, dropping duplicate boundary points
    t_out = np.concatenate([t1_arr, t2_arr[1:], t3_arr[1:]])
    y_out = np.concatenate([y1_arr, y2_arr[:, 1:], y3_arr[:, 1:]], axis=1)
    return t_out, y_out


def run_steady_state(params, cl=1000.0, n_beats=100, n_save=1,
                     ikr_block=0.0, y0=None, verbose=False):
    """
    Pace cell for n_beats beats at cycle length cl.
    Returns the last n_save beat(s).

    Parameters
    ----------
    params    : dict  — MALE or FEMALE conductance parameters
    cl        : float — cycle length (ms)
    n_beats   : int   — total beats
    n_save    : int   — beats to return (from end)
    ikr_block : float — IKr block fraction [0,1]
    y0        : array — initial state (defaults to standard ICs)

    Returns
    -------
    t_out : np.ndarray — time (ms), relative to first saved beat
    y_out : np.ndarray — state, shape (41, len(t_out))
    """
    if y0 is None:
        y0 = initial_conditions()

    y_cur = y0.copy()
    t_all = []
    y_all = []

    for beat_idx in range(n_beats):
        is_last = beat_idx >= n_beats - n_save
        t_b, y_b = pace_one_beat(
            y_cur, cl, params,
            ikr_block=ikr_block,
            dense=is_last          # dense output only for the beats we keep
        )
        y_cur = y_b[:, -1]

        if is_last:
            offset = (beat_idx - (n_beats - n_save)) * cl
            t_all.append(t_b + offset)
            y_all.append(y_b)

        if verbose and (beat_idx % 20 == 0 or beat_idx == n_beats - 1):
            vm = y_b[0, :]
            print(f"  beat {beat_idx+1}/{n_beats}: Vmax={np.max(vm):.1f} mV")

    t_out = np.concatenate(t_all)
    y_out = np.concatenate(y_all, axis=1)
    return t_out, y_out


def compute_apd(t, vm, threshold_frac=0.90):
    """
    Compute APD at threshold_frac repolarization for the last beat.
    Returns APD in ms, or NaN if no action potential detected.
    Uses the first upward crossing of -40 mV (stimulus-evoked upstroke),
    not the last, to avoid misidentification during EADs or secondary depolarizations.
    """
    # First upstroke: Vm crosses -40 mV going up (stimulus-evoked)
    up_idx = np.where((vm[:-1] < -40.0) & (vm[1:] >= -40.0))[0]
    if len(up_idx) == 0:
        return np.nan

    i_up = up_idx[0]
    t_up = t[i_up]
    Vmax  = np.max(vm[i_up:])
    Vrest = np.percentile(vm[:max(i_up, 1)], 5)  # 5th percentile as rest
    Vtarget = Vmax - threshold_frac * (Vmax - Vrest)

    vm_after = vm[i_up:]
    t_after  = t[i_up:]

    # Find first downward crossing after plateau (skip first 10 ms)
    i_skip = np.searchsorted(t_after, t_up + 10.0) - i_up
    if i_skip >= len(vm_after) - 1:
        return np.nan
    down_idx = np.where((vm_after[i_skip:-1] >= Vtarget) &
                        (vm_after[i_skip+1:] < Vtarget))[0]
    if len(down_idx) == 0:
        return np.nan

    i_down = down_idx[-1] + i_skip
    t_down = t_after[i_down]
    return t_down - t_up


def detect_repolarization_failure(t, vm, cl, failure_threshold=0.95):
    """
    Detect repolarization failure: cell does not repolarize to resting potential
    within failure_threshold * cl milliseconds after the action potential upstroke.

    This is the primary arrhythmia risk marker in the ORd model under IKr block —
    the model produces extreme APD prolongation rather than classic EAD oscillations.

    Parameters
    ----------
    t   : time array (ms)
    vm  : voltage array (mV)
    cl  : cycle length (ms)
    failure_threshold : fraction of CL; APD90 > failure_threshold*CL = failure

    Returns True if repolarization failure detected.
    """
    apd90 = compute_apd(t, vm, threshold_frac=0.90)
    if np.isnan(apd90):
        return True                          # Cell never repolarized at all
    return apd90 > failure_threshold * cl


def detect_critical_apd(t, vm, threshold_ms=500.0):
    """
    Returns True if APD90 exceeds a clinically dangerous threshold.
    500 ms corresponds to QTc ~500 ms — the standard clinical safety cutoff
    for drug-induced QT prolongation.
    """
    apd90 = compute_apd(t, vm, threshold_frac=0.90)
    if np.isnan(apd90):
        return True
    return apd90 > threshold_ms


def detect_ead(t, vm, min_depol=2.0, plateau_delay_ms=100.0):
    """
    Detect classic EAD oscillations (dV/dt-positive excursion during plateau).
    NOTE: The ORd model under simple tonic IKr block does not produce classic
    EAD humps — it produces extreme APD prolongation. Use detect_repolarization_failure
    or detect_critical_apd for arrhythmia risk assessment in this model.
    Returns True if a classic EAD oscillation is detected.
    """
    up_idx = np.where((vm[:-1] < -40.0) & (vm[1:] >= -40.0))[0]
    if len(up_idx) == 0:
        return False

    i_up = up_idx[-1]
    t_up = t[i_up]

    i_start = np.searchsorted(t, t_up + plateau_delay_ms)
    if i_start >= len(vm) - 2:
        return False

    vm_p = vm[i_start:]
    dv = np.diff(vm_p)

    local_min_idx = np.where((dv[:-1] < 0) & (dv[1:] >= 0))[0]
    for idx in local_min_idx:
        if vm_p[idx] < -65.0:
            continue
        up_seg = dv[idx:]
        end_up = np.where(up_seg < 0)[0]
        if len(end_up) == 0:
            amp = vm_p[-1] - vm_p[idx]
        else:
            amp = vm_p[idx + end_up[0]] - vm_p[idx]
        if amp >= min_depol:
            return True
    return False


def compute_triangulation(t, vm):
    """APD90 − APD30: marker of action potential triangulation."""
    apd30 = compute_apd(t, vm, threshold_frac=0.30)
    apd90 = compute_apd(t, vm, threshold_frac=0.90)
    if np.isnan(apd30) or np.isnan(apd90):
        return np.nan
    return apd90 - apd30
