"""
Numba-JIT-compiled ORd 2011 model for fast simulation.

Parameter vector indices (PARAM_*) must match the order in params_to_array().
All units: mV, ms, mM, mS/μF, pA/pF.
"""

import numpy as np
import numba
from numba import njit, float64

# ── Parameter vector indices ───────────────────────────────────────────────────
P_GNa   = 0
P_GNaL  = 1
P_Gto   = 2
P_GKr   = 3
P_GKs   = 4
P_GK1   = 5
P_Gncx  = 6
P_GKb   = 7
P_GpCa  = 8
P_PCa   = 9
P_PNa   = 10
P_PK    = 11
P_PNaK  = 12
P_GbNa  = 13
P_GbCa  = 14
P_Jupmax = 15
N_PARAMS = 16

# ── Physical / geometry constants ─────────────────────────────────────────────
_R   = 8314.0
_T   = 310.0
_F   = 96485.0

_L    = 0.01
_rad  = 0.0011
_vcell = 1000.0 * np.pi * _rad**2 * _L
_Ageo  = 2.0 * np.pi * _rad**2 + 2.0 * np.pi * _rad * _L
_Acap  = 2.0 * _Ageo
_vmyo  = 0.68  * _vcell
_vnsr  = 0.0552 * _vcell
_vjsr  = 0.0048 * _vcell
_vss   = 0.02  * _vcell

_Nao = 140.0
_Cao = 1.8
_Ko  = 5.4

# Ca²⁺ handling constants
_KmCaMK  = 0.15
_aCaMK   = 0.05
_bCaMK   = 0.00068
_CaMKo   = 0.05
_KmCaM   = 0.0015
_BSRmax  = 0.047
_KmBSR   = 0.00087
_BSLmax  = 1.124
_KmBSL   = 0.0087
_cmdnmax = 0.05
_kmcmdn  = 0.00238
_trpnmax = 0.07
_kmtrpn  = 0.0005
_csqnmax = 10.0
_kmcsqn  = 0.8
_bt      = 4.75

_EXP_LIM = 30.0


def params_to_array(param_dict):
    """Convert parameter dict to numpy array for numba."""
    p = np.zeros(N_PARAMS)
    p[P_GNa]    = param_dict["GNa"]
    p[P_GNaL]   = param_dict["GNaL"]
    p[P_Gto]    = param_dict["Gto"]
    p[P_GKr]    = param_dict["GKr"]
    p[P_GKs]    = param_dict["GKs"]
    p[P_GK1]    = param_dict["GK1"]
    p[P_Gncx]   = param_dict["Gncx"]
    p[P_GKb]    = param_dict["GKb"]
    p[P_GpCa]   = param_dict["GpCa"]
    p[P_PCa]    = param_dict["PCa"]
    p[P_PNa]    = param_dict["PNa"]
    p[P_PK]     = param_dict["PK"]
    p[P_PNaK]   = param_dict["PNaK"]
    p[P_GbNa]   = param_dict["GbNa"]
    p[P_GbCa]   = param_dict["GbCa"]
    p[P_Jupmax]  = param_dict["Jupmax"]
    return p


@njit(cache=True)
def _e(x):
    if x > _EXP_LIM:
        return np.exp(_EXP_LIM)
    if x < -_EXP_LIM:
        return np.exp(-_EXP_LIM)
    return np.exp(x)


@njit(cache=True)
def rhs_numba(t, y, p, Istim, ikr_block):
    """
    JIT-compiled RHS. p is a flat float64 array (params_to_array output).
    """
    dydt = np.zeros(41)

    # ── Unpack ────────────────────────────────────────────────────────────────
    Vm     = y[0]
    m      = y[1];  h  = y[2];  j  = y[3]
    mL     = y[4];  hL = y[5]
    a      = y[6];  iF = y[7];  iS = y[8]
    d      = y[9];  ff = y[10]; fs = y[11]
    fcaf   = y[12]; fcas = y[13]; jca = y[14]
    nca    = y[15]; ffp = y[16]; fcafp = y[17]
    xrf    = y[18]; xrs = y[19]
    xs1    = y[20]; xs2 = y[21]
    xk1    = y[22]
    CaMKt  = y[23]
    Nai    = max(y[24], 1e-4);  Nass = max(y[25], 1e-4)
    Ki     = max(y[26], 1e-4);  Kss  = max(y[27], 1e-4)
    Cai    = max(y[28], 1e-9);  Cass = max(y[29], 1e-9)
    Cansr  = max(y[30], 1e-9);  Cajsr = max(y[31], 1e-9)
    Jrelnp = y[32]; Jrelp = y[33]
    hLCaMK = y[34]
    iFCaMK = y[35]; iSCaMK = y[36]
    hCaMK  = y[37]; jCaMK  = y[38]
    aCaMK_g = y[39]

    # ── Clamp ─────────────────────────────────────────────────────────────────
    if Vm > 80.0:  Vm = 80.0
    if Vm < -150.0: Vm = -150.0

    vfrt  = Vm * _F / (_R * _T)
    vffrt = vfrt * _F
    ENa   = _R * _T / _F * np.log(_Nao / Nai)
    EK    = _R * _T / _F * np.log(_Ko  / Ki)
    EKs   = _R * _T / _F * np.log((_Ko + 0.01833 * _Nao) / (Ki + 0.01833 * Nai))

    # ── CaMKII ────────────────────────────────────────────────────────────────
    CaMKb = _CaMKo * (1.0 - CaMKt) / (1.0 + _KmCaM / Cass)
    CaMKa = CaMKb + CaMKt
    dydt[23] = _aCaMK * CaMKb * (CaMKb + CaMKt) - _bCaMK * CaMKt
    fCaMKp  = 1.0 / (1.0 + _KmCaMK / max(CaMKa, 1e-15))

    # ── INa ───────────────────────────────────────────────────────────────────
    mss  = 1.0 / (1.0 + _e(-(Vm + 39.57) / 9.871))
    tm   = 1.0 / (6.765 * _e((Vm + 11.64) / 34.77) +
                  8.552 * _e(-(Vm + 77.42) / 5.955) + 1e-15)
    hss  = 1.0 / (1.0 + _e((Vm + 82.90) / 6.086))
    th   = 1.0 / (1.432e-5 * _e(-(Vm + 1.196) / 6.285) +
                  6.149    * _e((Vm + 0.5096) / 20.27) + 1e-15)
    jss  = hss
    tj   = 2.038 + 1.0 / (0.02136 * _e(-(Vm + 100.6) / 8.281) +
                           0.3052  * _e((Vm + 0.9941) / 38.45) + 1e-15)
    hCaMKss = 1.0 / (1.0 + _e((Vm + 89.1) / 6.086))
    thCaMK  = 3.0 * th
    tjCaMK  = 1.46 * tj

    dydt[1]  = (mss     - m)     / tm
    dydt[2]  = (hss     - h)     / th
    dydt[3]  = (jss     - j)     / tj
    dydt[37] = (hCaMKss - hCaMK) / thCaMK
    dydt[38] = (jss     - jCaMK) / tjCaMK

    INa = p[P_GNa] * (Vm - ENa) * m**3 * \
          ((1.0 - fCaMKp) * h * j + fCaMKp * hCaMK * jCaMK)

    # ── INaL ──────────────────────────────────────────────────────────────────
    mLss    = 1.0 / (1.0 + _e(-(Vm + 42.85) / 5.264))
    hLss    = 1.0 / (1.0 + _e((Vm + 87.61) / 7.488))
    hLCaMKss = 1.0 / (1.0 + _e((Vm + 93.81) / 7.488))

    dydt[4]  = (mLss     - mL)     / tm
    dydt[5]  = (hLss     - hL)     / 200.0
    dydt[34] = (hLCaMKss - hLCaMK) / 600.0

    GNaL = p[P_GNaL] * (1.0 + 0.3 * fCaMKp)
    INaL = GNaL * (Vm - ENa) * mL * \
           ((1.0 - fCaMKp) * hL + fCaMKp * hLCaMK)

    # ── Ito ───────────────────────────────────────────────────────────────────
    ass_  = 1.0 / (1.0 + _e(-(Vm - 14.34) / 14.82))
    ta    = 1.0515 / (1.0 / (1.2089 * (1.0 + _e(-(Vm - 18.41) / 29.38)) + 1e-15) +
                      3.5   / (1.0 + _e((Vm + 100.0) / 29.38)) + 1e-15)
    iFss  = 1.0 / (1.0 + _e((Vm + 43.94) / 5.711))
    tiF   = 4.562 + 1.0 / (0.3933 * _e(-(Vm + 100.0) / 100.0) +
                             0.08004 * _e((Vm + 50.0) / 16.59) + 1e-15)
    tiS   = 23.62 + 1.0 / (0.001416 * _e(-(Vm + 96.52) / 59.05) +
                             1.78e-8  * _e((Vm + 114.1) / 8.079) + 1e-15)
    AiFss = 1.0 / (1.0 + _e((Vm - 213.6) / 151.2))
    AiSss = 1.0 - AiFss

    dydt[6]  = (ass_ - a)      / ta
    dydt[7]  = (iFss - iF)     / tiF
    dydt[8]  = (iFss - iS)     / tiS
    dydt[39] = (ass_ - aCaMK_g) / ta
    dydt[35] = (iFss - iFCaMK) / tiF
    dydt[36] = (iFss - iSCaMK) / tiS

    i_g    = AiFss * iF     + AiSss * iS
    iCaMK_g = AiFss * iFCaMK + AiSss * iSCaMK
    Ito = p[P_Gto] * (Vm - EK) * \
          ((1.0 - fCaMKp) * a * i_g + fCaMKp * aCaMK_g * iCaMK_g)

    # ── ICaL ──────────────────────────────────────────────────────────────────
    dss   = 1.0 / (1.0 + _e(-(Vm + 3.940) / 4.230))
    td    = 0.6 + 1.0 / (_e(-0.05 * (Vm + 6.0)) + _e(0.09 * (Vm + 14.0)) + 1e-15)
    fss   = 1.0 / (1.0 + _e((Vm + 19.58) / 3.696))
    tff   = 7.0 + 1.0 / (0.0045 * _e(-(Vm + 20.0) / 10.0) +
                           0.0045 * _e((Vm + 20.0) / 10.0) + 1e-15)
    tfs   = 1000.0 + 1.0 / (3.5e-5 * _e(-(Vm + 5.0) / 4.0) +
                              3.5e-5 * _e((Vm + 5.0) / 6.0) + 1e-15)
    Aff   = 0.6;  Afs = 0.4
    f_g   = Aff * ff + Afs * fs

    fcafss = fss
    tfcaf  = 7.0 + 1.0 / (0.04 * _e(-(Vm - 4.0) / 7.0) +
                            0.04 * _e((Vm - 4.0) / 7.0) + 1e-15)
    tfcas  = 100.0 + 1.0 / (0.00012 * _e(-Vm / 3.0) +
                              0.00012 * _e(Vm / 7.0) + 1e-15)
    Afcaf  = 0.3 + 0.6 / (1.0 + _e((Vm - 10.0) / 10.0))
    Afcas  = 1.0 - Afcaf
    fca    = Afcaf * fcaf + Afcas * fcas

    jcass  = 1.0 / (1.0 + _e((Vm + 18.08) / 2.747))
    tjca   = 75.0
    Kmn   = 0.002;  k2n = 1000.0
    km2n  = jca
    anca  = 1.0 / (km2n / k2n + (1.0 + Kmn / Cass)**4)

    dydt[9]  = (dss    - d)    / td
    dydt[10] = (fss    - ff)   / tff
    dydt[11] = (fss    - fs)   / tfs
    dydt[12] = (fcafss - fcaf) / tfcaf
    dydt[13] = (fcafss - fcas) / tfcas
    dydt[14] = (jcass  - jca)  / tjca
    dydt[15] = anca * k2n - nca * km2n
    dydt[16] = (fss    - ffp)   / (2.5 * tff)
    dydt[17] = (fcafss - fcafp) / (2.5 * tfcaf)

    z2 = 2.0 * vfrt;  z1 = vfrt
    if abs(z2) > 1e-6:
        PhiCaL = 4.0 * vffrt * (Cass * _e(z2) - 0.341 * _Cao) / (_e(z2) - 1.0 + 1e-15)
    else:
        PhiCaL = 4.0 * vffrt * (Cass - 0.341 * _Cao)
    if abs(z1) > 1e-6:
        PhiCaNa = vffrt * (0.75 * Nass * _e(z1) - 0.75 * _Nao) / (_e(z1) - 1.0 + 1e-15)
        PhiCaK  = vffrt * (0.75 * Kss  * _e(z1) - 0.75 * _Ko ) / (_e(z1) - 1.0 + 1e-15)
    else:
        PhiCaNa = vffrt * 0.75 * (Nass - _Nao)
        PhiCaK  = vffrt * 0.75 * (Kss  - _Ko)

    PCa   = p[P_PCa]
    PCaNa = p[P_PNa]
    PCaK  = p[P_PK]
    PCap   = 1.1    * PCa;   PCaNap = 0.00125 * PCap;  PCaKp = 3.574e-4 * PCap

    fICaLp = 1.0 / (1.0 + _KmCaMK / max(CaMKa, 1e-15))
    fp_g   = (1.0 - fICaLp) * f_g  + fICaLp * (Aff * ffp + Afs * fs)
    fcap   = (1.0 - fICaLp) * fca  + fICaLp * (Afcaf * fcafp + Afcas * fcas)
    on     = 1.0 - nca

    ICaL  = ((1.0 - fICaLp) * PCa   * PhiCaL  * d * f_g  * fca  * on +
              fICaLp         * PCap  * PhiCaL  * d * fp_g * fcap * on)
    ICaNa = ((1.0 - fICaLp) * PCaNa * PhiCaNa * d * f_g  * fca  * on +
              fICaLp         * PCaNap * PhiCaNa * d * fp_g * fcap * on)
    ICaK  = ((1.0 - fICaLp) * PCaK  * PhiCaK  * d * f_g  * fca  * on +
              fICaLp         * PCaKp * PhiCaK  * d * fp_g * fcap * on)

    # ── IKr ───────────────────────────────────────────────────────────────────
    xrss  = 1.0 / (1.0 + _e(-(Vm + 8.337) / 6.789))
    txrf  = 12.98 + 1.0 / (0.3652  * _e((Vm - 31.66) / 3.869) +
                             4.123e-5 * _e(-(Vm - 47.78) / 20.38) + 1e-15)
    txrs  = 1.865 + 1.0 / (0.06629  * _e((Vm - 34.70) / 7.355) +
                             1.128e-5 * _e(-(Vm - 29.74) / 25.94) + 1e-15)
    Axrf  = 1.0 / (1.0 + _e((Vm + 54.81) / 38.21))
    xr    = Axrf * xrf + (1.0 - Axrf) * xrs
    rkr   = 1.0 / ((1.0 + _e((Vm + 55.0) / 75.0)) *
                   (1.0 + _e((Vm - 10.0) / 30.0)))

    dydt[18] = (xrss - xrf) / txrf
    dydt[19] = (xrss - xrs) / txrs

    GKr_eff = p[P_GKr] * ((_Ko / 5.4)**0.5) * (1.0 - ikr_block)
    IKr  = GKr_eff * (Vm - EK) * xr * rkr

    # ── IKs ───────────────────────────────────────────────────────────────────
    xs1ss = 1.0 / (1.0 + _e(-(Vm + 11.60) / 8.932))
    txs1  = 817.3 + 1.0 / (2.326e-4 * _e((Vm + 48.28) / 17.80) +
                             0.001292 * _e(-(Vm + 210.0) / 230.0) + 1e-15)
    txs2  = 1.0 / (0.01   * _e((Vm - 50.0) / 20.0) +
                   0.0193  * _e(-(Vm + 66.54) / 31.0) + 1e-15)
    KsCa  = 1.0 + 0.6 / (1.0 + (3.8e-5 / Cai)**1.4)

    dydt[20] = (xs1ss - xs1) / txs1
    dydt[21] = (xs1ss - xs2) / txs2

    IKs = p[P_GKs] * KsCa * xs1 * xs2 * (Vm - EKs)

    # ── IK1 ───────────────────────────────────────────────────────────────────
    xk1ss = 1.0 / (1.0 + _e(-(Vm + 2.5538 * _Ko + 144.59) /
                               (1.5692 * _Ko + 3.8115)))
    txk1  = 122.2 / (_e(-(Vm + 127.2) / 20.36) +
                     _e((Vm + 236.8) / 69.33) + 1e-15)
    rkk1  = 1.0 / (1.0 + _e((Vm + 105.8 - 2.6 * _Ko) / 9.493))

    dydt[22] = (xk1ss - xk1) / txk1

    GK1_eff = p[P_GK1] * ((_Ko / 5.0)**0.5)
    IK1  = GK1_eff * rkk1 * xk1 * (Vm - EK)

    # ── IKb ───────────────────────────────────────────────────────────────────
    IKb  = p[P_GKb] / (1.0 + _e(-(Vm - 14.48) / 18.34)) * (Vm - EK)

    # ── INaCa (Shannon-Bers) ──────────────────────────────────────────────────
    KmCa  = 1.25e-4;  KmNai = 12.3;  KmNao = 87.5;  KmCao = 1.3
    eta   = 0.35;     ksat  = 0.1

    def _ncx_i(Nai_, Cai_, frac):
        num = _e(eta * vfrt) * Nai_**3 * _Cao - _e((eta - 1.0) * vfrt) * _Nao**3 * Cai_
        denom = (KmNao**3 + _Nao**3) * (KmCao + _Cao) * \
                (1.0 + ksat * _e((eta - 1.0) * vfrt))
        allo  = 1.0 / (1.0 + (KmCa / Cai_)**2)
        return frac * p[P_Gncx] * allo * num / denom

    INaCa_i  = _ncx_i(Nai,  Cai,  0.8)
    INaCa_ss = _ncx_i(Nass, Cass, 0.2)
    INaCa    = INaCa_i + INaCa_ss

    # ── INaK (Shannon-Bers) ───────────────────────────────────────────────────
    sigma  = (_e(_Nao / 67.3) - 1.0) / 7.0
    fnak   = 1.0 / (1.0 + 0.1245 * _e(-0.1 * vfrt) +
                    0.0365 * sigma * _e(-vfrt))
    INaK   = p[P_PNaK] * fnak * \
             _Ko / (_Ko + 1.5) * \
             Nai**1.5 / (Nai**1.5 + 10.0**1.5)

    # ── Background + sarcolemmal pump ─────────────────────────────────────────
    IbNa = p[P_GbNa] * (Vm - ENa)
    ECa_v = 0.5 * _R * _T / _F * np.log(_Cao / Cai)
    IbCa = p[P_GbCa] * (Vm - ECa_v)
    IpCa = p[P_GpCa] * Cai / (5.0e-4 + Cai)

    # ── Ca²⁺ handling ────────────────────────────────────────────────────────
    Jupmax  = p[P_Jupmax]
    f_SERCA = 1.0 + 0.16 * fCaMKp
    Jup_up  = f_SERCA * 0.004375 * Cai / (Cai + 9.2e-4)
    Jup_dn  = 0.004375 * Cansr / 15.0
    Jup     = (Jup_up - Jup_dn) * Jupmax / 0.004375

    Jtr     = (Cansr - Cajsr) / 100.0

    _Jrel_scale = _Acap / (2.0 * _F * _vjsr)
    Jrel_inf_np = -ICaL * _bt * _Jrel_scale / (1.0 + (1.5 / Cajsr)**8)
    Jrel_inf_p  = Jrel_inf_np * 1.25
    tau_rel_raw = _bt / (1.0 + 0.0123 / Cajsr)
    tau_rel     = tau_rel_raw if tau_rel_raw > 0.005 else 0.005

    dydt[32] = (Jrel_inf_np - Jrelnp) / tau_rel
    dydt[33] = (Jrel_inf_p  - Jrelp)  / tau_rel

    Jrel    = (1.0 - fCaMKp) * Jrelnp + fCaMKp * Jrelp
    Jdiff   = (Cass - Cai)  / 0.2
    JdiffNa = (Nass - Nai)  / 2.0
    JdiffK  = (Kss  - Ki)   / 2.0

    # ── Ion concentration ODEs ────────────────────────────────────────────────
    dydt[24] = -(INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + IbNa) * \
               _Acap / (_F * _vmyo) + JdiffNa * _vss / _vmyo
    dydt[25] = -(ICaNa + 3.0 * INaCa_ss) * _Acap / (_F * _vss) - JdiffNa

    dydt[26] = -(Ito + IKr + IKs + IK1 + IKb - 2.0 * INaK + ICaK + Istim) * \
               _Acap / (_F * _vmyo) + JdiffK * _vss / _vmyo
    dydt[27] = -ICaK * _Acap / (_F * _vss) - JdiffK

    Bcai  = 1.0 / (1.0 + _cmdnmax * _kmcmdn / (_kmcmdn + Cai)**2 +
                   _trpnmax * _kmtrpn / (_kmtrpn + Cai)**2)
    dydt[28] = Bcai * (-(IbCa + IpCa - 2.0 * INaCa_i) * _Acap / (2.0 * _F * _vmyo) -
               Jup * _vnsr / _vmyo + Jdiff * _vss / _vmyo)

    Bcass = 1.0 / (1.0 + _BSRmax * _KmBSR / (_KmBSR + Cass)**2 +
                   _BSLmax * _KmBSL / (_KmBSL + Cass)**2)
    dydt[29] = Bcass * (-(ICaL - 2.0 * INaCa_ss) * _Acap / (2.0 * _F * _vss) +
               Jrel * _vjsr / _vss - Jdiff)

    dydt[30] = Jup - Jtr * _vjsr / _vnsr

    Bcajsr = 1.0 / (1.0 + _csqnmax * _kmcsqn / (_kmcsqn + Cajsr)**2)
    dydt[31] = Bcajsr * (Jtr - Jrel)

    # ── Membrane voltage ──────────────────────────────────────────────────────
    Itot = (INa + INaL + Ito + ICaL + ICaNa + ICaK +
            IKr + IKs + IK1 + IKb +
            INaCa + INaK + IpCa + IbNa + IbCa + Istim)
    dydt[0] = -Itot

    return dydt
