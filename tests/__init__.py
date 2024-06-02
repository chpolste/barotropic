# np.isclose and np.allclose must be given an appropriate value of atol for
# different fields. Also, pyspharm uses single-precision floats, so the values
# must account for the limited accuracy possible even when comparing doubles.
ATOL_WIND = 1e-4 # m/s
ATOL_PV   = 1e-10 # 1/s
ATOL_VO   = 1e-10 # 1/s
ATOL_PSI  = 1e+1

