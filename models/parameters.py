"""
O'Hara-Rudy (ORd) 2011 model parameters with sex-specific variants.

Male parameters: original ORd 2011 (epicardial, endocardial, midmyocardial)
Female parameters: adjusted based on:
  - Kurokawa et al. (2016) JAHA: GKs female ~50-60% of male
  - Bett et al. (2006) AJP: Ito female ~60% of male
  - Stroud et al. (2015): similar INa, IKr between sexes
  - Du et al. (2015): slightly higher IKr in female (minor)

Reference: O'Hara T, et al. PLoS Comput Biol. 2011;7(5):e1002061.
"""

import numpy as np

# ─── Physical constants ────────────────────────────────────────────────────────
R = 8314.0       # J/kmol/K
T = 310.0        # K (37 °C)
F = 96485.0      # C/mol

# ─── Cell geometry ─────────────────────────────────────────────────────────────
L    = 0.01      # cm
rad  = 0.0011    # cm
vcell = 1000 * np.pi * rad**2 * L   # μL
Ageo  = 2 * np.pi * rad**2 + 2 * np.pi * rad * L
Acap  = 2 * Ageo
vmyo  = 0.68 * vcell
vnsr  = 0.0552 * vcell
vjsr  = 0.0048 * vcell
vss   = 0.02 * vcell

# ─── External ion concentrations (mM) ─────────────────────────────────────────
Nao  = 140.0
Cao  = 1.8
Ko   = 5.4

# ─── Baseline (male) maximal conductances / permeabilities ────────────────────
# These correspond to the epicardial ORd formulation
MALE = {
    "GNa":   75.0,          # mS/μF   fast Na
    "GNaL":  0.0075,        # mS/μF   late Na
    "Gto":   0.02,          # mS/μF   transient outward K (epi)
    "GKr":   0.046,         # mS/μF   rapid delayed rectifier
    "GKs":   0.0034,        # mS/μF   slow delayed rectifier
    "GK1":   0.1908,        # mS/μF   inward rectifier
    "Gncx":  3500.0,        # scaling for Shannon-Bers NCX (tuned to give ~-0.3 pA/pF at rest)
    "GKb":   0.003,         # mS/μF   background K
    "GpCa":  0.0005,        # mM/ms   sarcolemmal Ca pump
    "PCa":   0.0001,        # cm/s    ICaL permeability
    "PNa":   0.75e-8,       # cm/s    ICaNa permeability
    "PK":    2.75e-7,       # cm/s    ICaK permeability
    # Na-K pump: Shannon-Bers Hill-function formulation
    "PNaK":  1.362,         # pA/pF   max rate (~0.7 pA/pF at rest)
    # Background
    "GbNa":  3.75e-4,       # mS/μF
    "GbCa":  5.9e-4,        # mS/μF
    # SERCA
    "Jupmax": 0.006375,     # mM/ms
}

# Female parameters: apply scaling factors derived from published literature.
# Primary driver: IKs reduction (Kurokawa et al. 2016, best-established).
# Ito and IKr changes are omitted: Bett 2006 was in guinea pig (not human),
# Du 2015 IKr upregulation is controversial and negligible in magnitude.
# Using IKs alone reproduces the observed ~5 ms female QTc prolongation.
_FEMALE_SCALE = {
    "GKs":  0.55,   # −45% IKs  (Kurokawa et al. 2016)
}

FEMALE = dict(MALE)
for key, scale in _FEMALE_SCALE.items():
    FEMALE[key] = MALE[key] * scale

# ─── CaMKII parameters ────────────────────────────────────────────────────────
KmCaMK   = 0.15
aCaMK    = 0.05
bCaMK    = 0.00068
CaMKo    = 0.05
KmCaM    = 0.0015

# ─── Ca²⁺ handling parameters ─────────────────────────────────────────────────
BSRmax  = 0.047
KmBSR   = 0.00087
BSLmax  = 1.124
KmBSL   = 0.0087
cmdnmax = 0.05
kmcmdn  = 0.00238
trpnmax = 0.07
kmtrpn  = 0.0005
csqnmax = 10.0
kmcsqn  = 0.8

# ─── RyR parameters ───────────────────────────────────────────────────────────
bt      = 4.75
a_rel   = 0.5 * bt
Jrel_inf_mult = 1.0   # can be modified for disease
