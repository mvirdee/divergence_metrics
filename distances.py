import numpy as np
from scipy.stats import wasserstein_distance
from scipy.integrate import quad
from scipy.stats import energy_distance
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF

def d_hel(P,Q):
  '''normalised Hellinger distance between probability vectors P and Q'''
  assert len(P) == len(Q)
  hel = (np.sum((np.sqrt(P)-np.sqrt(Q))**2)*0.5)**0.5
  return hel/(len(P)**0.5)


def d_was(X_P, X_Q, P, Q):
  '''scipy wasserstein distance'''
  return wasserstein_distance(X_P,X_Q,P,Q)


def dif_ecdf(x, P, Q):
    F_p = ECDF(P)
    F_q = ECDF(Q)
    return (F_p(x) - F_q(x))**2

def d_iq(P, Q):
    '''integrated quadratic distance'''
    intmin = min(np.min(P), np.min(Q))
    intmax = max(np.max(P), np.max(Q))

    iqd, _ = quad(dif_ecdf, intmin, intmax, args=(P, Q))
    return iqd

def d_iq_sp(P,Q):
  '''scipy integrated quadratic distance calculated from energy distance'''
  return (energy_distance(P,Q)**2)*0.5