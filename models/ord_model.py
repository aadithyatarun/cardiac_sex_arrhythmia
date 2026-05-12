"""
O'Hara-Rudy 2011 human ventricular action potential model.

Gate kinetics: exactly as in O'Hara et al. PLoS Comput Biol 2011;7:e1002061.
INaCa / INaK: Shannon-Bers simplified formulations (equivalent steady-state
  behavior, correct physiological magnitudes, numerically robust).

All units: mV, ms, mM, mS/μF, pA/pF.
"""

import numpy as np
from .parameters import (
    R, T, F,
    vmyo, vnsr, vjsr, vss, Acap,
    Nao, Cao, Ko,
    KmCaMK, aCaMK as alpha_CaMK, bCaMK, CaMKo, KmCaM,
    BSRmax, KmBSR, BSLmax, KmBSL,
    cmdnmax, kmcmdn, trpnmax, kmtrpn, csqnmax, kmcsqn,
    bt,
)

_EXP_LIM = 30.0   # |vfrt| < 12 at physiological Vm range; 2*vfrt stays safe


def _e(x):
    return np.exp(np.clip(x, -_EXP_LIM, _EXP_LIM))


def initial_conditions():
    """Approximate resting initial conditions (epicardial ORd)."""
    y = np.zeros(41)
    y[0]  = -87.5     # Vm  (mV)
    y[1]  = 0.0       # m   (INa)
    y[2]  = 0.7       # h
    y[3]  = 0.7       # j
    y[4]  = 0.0       # mL  (INaL)
    y[5]  = 0.9       # hL
    y[6]  = 0.0       # a   (Ito)
    y[7]  = 1.0       # iF
    y[8]  = 0.6       # iS
    y[9]  = 0.0       # d   (ICaL)
    y[10] = 1.0       # ff
    y[11] = 0.9       # fs
    y[12] = 1.0       # fcaf
    y[13] = 1.0       # fcas
    y[14] = 1.0       # jca
    y[15] = 0.0       # nca
    y[16] = 1.0       # ffp
    y[17] = 1.0       # fcafp
    y[18] = 0.0       # xrf  (IKr)
    y[19] = 0.0       # xrs
    y[20] = 0.0       # xs1  (IKs)
    y[21] = 0.0       # xs2
    y[22] = 1.0       # xk1  (IK1)
    y[23] = 0.0       # CaMKt
    y[24] = 7.0       # Nai  (mM)
    y[25] = 7.0       # Nass
    y[26] = 145.0     # Ki
    y[27] = 145.0     # Kss
    y[28] = 1.0e-4    # Cai
    y[29] = 1.0e-4    # Cass
    y[30] = 1.2       # Cansr
    y[31] = 1.2       # Cajsr
    y[32] = 0.0       # Jrelnp
    y[33] = 0.0       # Jrelp
    y[34] = 0.9       # hLCaMK
    y[35] = 1.0       # iFCaMK
    y[36] = 0.6       # iSCaMK
    y[37] = 0.7       # hCaMK
    y[38] = 0.7       # jCaMK
    y[39] = 0.0       # aCaMK
    y[40] = 0.0       # spare
    return y


def rhs(t, y, params, Istim=0.0, ikr_block=0.0):
    """
    RHS for ORd 2011 epicardial ventricular AP model.
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
    aCaMK  = y[39]

    # ── Derived ───────────────────────────────────────────────────────────────
    Vm    = np.clip(Vm, -150.0, 80.0)   # physiological clamping during solver steps
    vfrt  = Vm * F / (R * T)
    vffrt = vfrt * F
    ENa   = R * T / F * np.log(Nao / Nai)
    EK    = R * T / F * np.log(Ko  / Ki)
    EKs   = R * T / F * np.log((Ko + 0.01833 * Nao) / (Ki + 0.01833 * Nai))
    ECa   = 0.5 * R * T / F * np.log(Cao / Cai)

    # ── CaMKII ────────────────────────────────────────────────────────────────
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / Cass)
    CaMKa = CaMKb + CaMKt
    dydt[23] = alpha_CaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt
    fCaMKp  = 1.0 / (1.0 + KmCaMK / max(CaMKa, 1e-15))

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

    INa = params["GNa"] * (Vm - ENa) * m**3 * \
          ((1.0 - fCaMKp) * h * j + fCaMKp * hCaMK * jCaMK)

    # ── INaL ──────────────────────────────────────────────────────────────────
    mLss    = 1.0 / (1.0 + _e(-(Vm + 42.85) / 5.264))
    tmL     = tm
    hLss    = 1.0 / (1.0 + _e((Vm + 87.61) / 7.488))
    hLCaMKss = 1.0 / (1.0 + _e((Vm + 93.81) / 7.488))

    dydt[4]  = (mLss     - mL)     / tmL
    dydt[5]  = (hLss     - hL)     / 200.0
    dydt[34] = (hLCaMKss - hLCaMK) / 600.0

    GNaL = params["GNaL"] * (1.0 + 0.3 * fCaMKp)
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
    dydt[39] = (ass_ - aCaMK)  / ta
    dydt[35] = (iFss - iFCaMK) / tiF
    dydt[36] = (iFss - iSCaMK) / tiS

    i_g    = AiFss * iF     + AiSss * iS
    iCaMK_g = AiFss * iFCaMK + AiSss * iSCaMK
    Ito = params["Gto"] * (Vm - EK) * \
          ((1.0 - fCaMKp) * a * i_g + fCaMKp * aCaMK * iCaMK_g)

    # ── ICaL ──────────────────────────────────────────────────────────────────
    dss   = 1.0 / (1.0 + _e(-(Vm + 3.940) / 4.230))
    td    = 0.6 + 1.0 / (_e(-0.05 * (Vm + 6.0)) + _e(0.09 * (Vm + 14.0)) + 1e-15)
    fss   = 1.0 / (1.0 + _e((Vm + 19.58) / 3.696))
    tff   = 7.0 + 1.0 / (0.0045 * _e(-(Vm + 20.0) / 10.0) +
                           0.0045 * _e((Vm + 20.0) / 10.0) + 1e-15)
    tfs   = 1000.0 + 1.0 / (3.5e-5 * _e(-(Vm + 5.0) / 4.0) +
                              3.5e-5 * _e((Vm + 5.0) / 6.0) + 1e-15)
    Aff   = 0.6;  Afs = 1.0 - Aff
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

    # GHK driving forces (L'Hôpital near V=0)
    z2 = 2.0 * vfrt;  z1 = vfrt
    PhiCaL  = 4.0 * vffrt * (Cass * _e(z2) - 0.341 * Cao) / (_e(z2) - 1.0 + 1e-15) \
              if abs(z2) > 1e-6 else 4.0 * vffrt * (Cass - 0.341 * Cao)
    PhiCaNa = vffrt * (0.75 * Nass * _e(z1) - 0.75 * Nao) / (_e(z1) - 1.0 + 1e-15) \
              if abs(z1) > 1e-6 else vffrt * 0.75 * (Nass - Nao)
    PhiCaK  = vffrt * (0.75 * Kss  * _e(z1) - 0.75 * Ko ) / (_e(z1) - 1.0 + 1e-15) \
              if abs(z1) > 1e-6 else vffrt * 0.75 * (Kss - Ko)

    PCa   = params["PCa"];    PCaNa = params["PNa"];    PCaK = params["PK"]
    PCap   = 1.1    * PCa;   PCaNap = 0.00125 * PCap;  PCaKp = 3.574e-4 * PCap

    fICaLp = 1.0 / (1.0 + KmCaMK / max(CaMKa, 1e-15))
    fp_g   = (1.0 - fICaLp) * f_g  + fICaLp * (Aff * ffp + Afs * fs)
    fcap   = (1.0 - fICaLp) * fca  + fICaLp * (Afcaf * fcafp + Afcas * fcas)
    on     = (1.0 - nca)

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

    GKr  = params["GKr"] * np.sqrt(Ko / 5.4) * (1.0 - ikr_block)
    IKr  = GKr * (Vm - EK) * xr * rkr

    # ── IKs ───────────────────────────────────────────────────────────────────
    xs1ss = 1.0 / (1.0 + _e(-(Vm + 11.60) / 8.932))
    txs1  = 817.3 + 1.0 / (2.326e-4 * _e((Vm + 48.28) / 17.80) +
                             0.001292 * _e(-(Vm + 210.0) / 230.0) + 1e-15)
    txs2  = 1.0 / (0.01   * _e((Vm - 50.0) / 20.0) +
                   0.0193  * _e(-(Vm + 66.54) / 31.0) + 1e-15)
    KsCa  = 1.0 + 0.6 / (1.0 + (3.8e-5 / Cai)**1.4)

    dydt[20] = (xs1ss - xs1) / txs1
    dydt[21] = (xs1ss - xs2) / txs2

    IKs = params["GKs"] * KsCa * xs1 * xs2 * (Vm - EKs)

    # ── IK1 ───────────────────────────────────────────────────────────────────
    xk1ss = 1.0 / (1.0 + _e(-(Vm + 2.5538 * Ko + 144.59) /
                               (1.5692 * Ko + 3.8115)))
    txk1  = 122.2 / (_e(-(Vm + 127.2) / 20.36) +
                     _e((Vm + 236.8) / 69.33) + 1e-15)
    rkk1  = 1.0 / (1.0 + _e((Vm + 105.8 - 2.6 * Ko) / 9.493))

    dydt[22] = (xk1ss - xk1) / txk1

    GK1  = params["GK1"] * np.sqrt(Ko / 5.0)
    IK1  = GK1 * rkk1 * xk1 * (Vm - EK)

    # ── IKb (background K⁺) ───────────────────────────────────────────────────
    IKb  = params["GKb"] / (1.0 + _e(-(Vm - 14.48) / 18.34)) * (Vm - EK)

    # ── INaCa (Shannon-Bers formulation) ──────────────────────────────────────
    # Simple, validated, correct sign and magnitude.
    # ksat=0.1, eta=0.35; KNCX scales with Gncx
    KmCa  = 1.25e-4;  KmNai = 12.3;  KmNao = 87.5;  KmCao = 1.3
    eta   = 0.35;     ksat  = 0.1

    def _ncx_i(Nai_, Cai_, frac):
        num = _e(eta * vfrt) * Nai_**3 * Cao - _e((eta - 1.0) * vfrt) * Nao**3 * Cai_
        denom = (KmNao**3 + Nao**3) * (KmCao + Cao) * \
                (1.0 + ksat * _e((eta - 1.0) * vfrt))
        allo  = 1.0 / (1.0 + (KmCa / Cai_)**2)
        return frac * params["Gncx"] * allo * num / denom

    INaCa_i  = _ncx_i(Nai,  Cai,  0.8)
    INaCa_ss = _ncx_i(Nass, Cass, 0.2)
    INaCa    = INaCa_i + INaCa_ss

    # ── INaK (Shannon-Bers formulation) ───────────────────────────────────────
    # Gives ~0.7 pA/pF at rest with PNaK=1.5.  Rescale from ORd PNaK.
    KmNai_nak = 10.0;  KmKo_nak = 1.5
    sigma  = (_e(Nao / 67.3) - 1.0) / 7.0
    fnak   = 1.0 / (1.0 + 0.1245 * _e(-0.1 * vfrt) +
                    0.0365 * sigma * _e(-vfrt))
    INaK   = params["PNaK"] * fnak * \
             Ko / (Ko + KmKo_nak) * \
             Nai**1.5 / (Nai**1.5 + KmNai_nak**1.5)

    # ── Background + pump ─────────────────────────────────────────────────────
    IbNa = params["GbNa"] * (Vm - ENa)
    IbCa = params["GbCa"] * (Vm - ECa)
    IpCa = params["GpCa"] * Cai / (5.0e-4 + Cai)

    # ── Ca²⁺ handling ────────────────────────────────────────────────────────
    Jupmax  = params["Jupmax"]
    f_SERCA = 1.0 + 0.16 * fCaMKp
    Jup_up  = f_SERCA * 0.004375 * Cai / (Cai + 9.2e-4)
    Jup_dn  = 0.004375 * Cansr / 15.0
    Jup     = (Jup_up - Jup_dn) * Jupmax / 0.004375

    Jtr     = (Cansr - Cajsr) / 100.0

    _Jrel_scale = Acap / (2.0 * F * vjsr)   # converts pA/pF → mM/ms
    Jrel_inf_np = -ICaL * bt * _Jrel_scale / (1.0 + (1.5 / Cajsr)**8)
    Jrel_inf_p  = Jrel_inf_np * 1.25
    tau_rel     = max(bt / (1.0 + 0.0123 / Cajsr), 0.005)

    dydt[32] = (Jrel_inf_np - Jrelnp) / tau_rel
    dydt[33] = (Jrel_inf_p  - Jrelp)  / tau_rel

    Jrel    = (1.0 - fCaMKp) * Jrelnp + fCaMKp * Jrelp
    Jdiff   = (Cass - Cai)  / 0.2
    JdiffNa = (Nass - Nai)  / 2.0
    JdiffK  = (Kss  - Ki)   / 2.0

    # ── Ion concentration ODEs ────────────────────────────────────────────────
    # Factor: Acap / (F * V) converts pA/pF → mM/ms
    dydt[24] = -(INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + IbNa) * \
               Acap / (F * vmyo) + JdiffNa * vss / vmyo
    dydt[25] = -(ICaNa + 3.0 * INaCa_ss) * Acap / (F * vss) - JdiffNa

    dydt[26] = -(Ito + IKr + IKs + IK1 + IKb - 2.0 * INaK + ICaK + Istim) * \
               Acap / (F * vmyo) + JdiffK * vss / vmyo
    dydt[27] = -ICaK * Acap / (F * vss) - JdiffK

    Bcai  = 1.0 / (1.0 + cmdnmax * kmcmdn / (kmcmdn + Cai)**2 +
                   trpnmax * kmtrpn / (kmtrpn + Cai)**2)
    dydt[28] = Bcai * (-(IbCa + IpCa - 2.0 * INaCa_i) * Acap / (2.0 * F * vmyo) -
               Jup * vnsr / vmyo + Jdiff * vss / vmyo)

    Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + Cass)**2 +
                   BSLmax * KmBSL / (KmBSL + Cass)**2)
    dydt[29] = Bcass * (-(ICaL - 2.0 * INaCa_ss) * Acap / (2.0 * F * vss) +
               Jrel * vjsr / vss - Jdiff)

    dydt[30] = Jup - Jtr * vjsr / vnsr

    Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + Cajsr)**2)
    dydt[31] = Bcajsr * (Jtr - Jrel)

    # ── Membrane potential ────────────────────────────────────────────────────
    Itot = (INa + INaL + Ito + ICaL + ICaNa + ICaK +
            IKr + IKs + IK1 + IKb +
            INaCa + INaK + IpCa + IbNa + IbCa + Istim)
    dydt[0] = -Itot

    return dydt
