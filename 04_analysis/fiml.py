"""Full-information maximum likelihood (FIML) for the baseline's parallel-mediation path model.

Variables are ordered [x (5), mediators (6), outcome (1)]. The covariates x are random
(as lavaan does with fixed.x = FALSE), so rows with a missing covariate are kept.
Robust standard errors use a sandwich: inverse Hessian, outer product of casewise scores,
inverse Hessian (the MLR idea). Casewise scores are analytic in (mu, Sigma); only the
small model Jacobians and the Hessian are taken by finite differences.
"""
import numpy as np
from scipy.optimize import minimize

Q, P_MED = 5, 6
P_Y = P_MED + 1


class PathModel:
    def __init__(self, Z, free_med_cov=False):
        self.Z = np.asarray(Z, float)
        self.free = free_med_cov
        self.n, self.m = self.Z.shape
        pats = {}
        for i, row in enumerate(~np.isnan(self.Z)):
            pats.setdefault(row.tobytes(), []).append(i)
        self.groups = [(np.frombuffer(k, dtype=bool), np.array(v)) for k, v in pats.items()]
        self.n_psi = 21 if free_med_cov else 6
        self.n_par = Q + 15 + P_Y + 30 + Q + 6 + self.n_psi + 1

    # ---- parameters -> mean vector and covariance matrix ----
    def unpack(self, th):
        i = 0
        mx = th[i:i + Q]; i += Q
        Lx = np.zeros((Q, Q)); Lx[np.tril_indices(Q)] = th[i:i + 15]; i += 15
        G0 = th[i:i + P_Y]; i += P_Y
        Gx = np.zeros((P_Y, Q)); Gx[:6] = th[i:i + 30].reshape(6, Q); i += 30
        Gx[6] = th[i:i + Q]; i += Q
        b = th[i:i + 6]; i += 6
        L = np.zeros((6, 6))
        if self.free:
            L[np.tril_indices(6)] = th[i:i + 21]; i += 21
        else:
            L[np.diag_indices(6)] = th[i:i + 6]; i += 6
        Psi = np.zeros((P_Y, P_Y)); Psi[:6, :6] = L @ L.T; Psi[6, 6] = th[i] ** 2
        B = np.zeros((P_Y, P_Y)); B[6, :6] = b
        IA = np.linalg.inv(np.eye(P_Y) - B)
        Sxx = Lx @ Lx.T; C = IA @ Gx
        mu = np.concatenate([mx, IA @ G0 + C @ mx])
        S = np.block([[Sxx, Sxx @ C.T], [C @ Sxx, C @ Sxx @ C.T + IA @ Psi @ IA.T]])
        return mu, S, Gx, b

    def index_a(self, j):            # coefficient of x[0] (the treatment) in mediator j
        return Q + 15 + P_Y + j * Q

    def index_c(self):               # direct effect of x[0] on the outcome
        return Q + 15 + P_Y + 30

    def index_b(self, j):
        return Q + 15 + P_Y + 30 + Q + j

    # ---- likelihood ----
    def nll(self, th):
        mu, S, _, _ = self.unpack(th)
        tot = 0.0
        for o, ix in self.groups:
            try:
                c = np.linalg.cholesky(S[np.ix_(o, o)])
            except np.linalg.LinAlgError:
                return 1e12
            r = self.Z[np.ix_(ix, np.where(o)[0])] - mu[o]
            z = np.linalg.solve(c, r.T)
            tot += 0.5 * (len(ix) * (2 * np.log(np.diag(c)).sum() + o.sum() * np.log(2 * np.pi)) + (z ** 2).sum())
        return tot

    def start(self):
        X, Y = self.Z[:, :Q], self.Z[:, Q:]
        mx = np.nanmean(X, 0)
        Sx = np.ma.cov(np.ma.masked_invalid(X), rowvar=False).filled(0) + 1e-6 * np.eye(Q)
        th = list(mx) + list(np.linalg.cholesky(Sx)[np.tril_indices(Q)])
        sd = np.nanstd(Y, 0)
        th += list(np.nanmean(Y, 0)) + [0.0] * 35 + [0.0] * 6
        th += list(np.diag(sd[:6])[np.tril_indices(6)]) if self.free else list(sd[:6])
        return np.array(th + [sd[6]])

    def fit(self):
        r = minimize(self.nll, self.start(), method="L-BFGS-B", options={"maxiter": 50000, "maxfun": 10 ** 7})
        r2 = minimize(self.nll, r.x, method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
        self.theta = r2.x if r2.fun <= r.fun else r.x
        self.fun = min(r.fun, r2.fun)
        return self

    # ---- robust standard errors ----
    def _model_jac(self, th, h=1e-6):
        mu0, S0, _, _ = self.unpack(th)
        Dmu = np.zeros((self.m, self.n_par)); DS = np.zeros((self.m, self.m, self.n_par))
        for k in range(self.n_par):
            e = np.zeros(self.n_par); e[k] = h
            mp, Sp, _, _ = self.unpack(th + e); mm, Sm, _, _ = self.unpack(th - e)
            Dmu[:, k] = (mp - mm) / (2 * h); DS[:, :, k] = (Sp - Sm) / (2 * h)
        return Dmu, DS

    def scores(self, th):
        mu, S, _, _ = self.unpack(th)
        Dmu, DS = self._model_jac(th)
        out = np.zeros((self.n, self.n_par))
        for o, ix in self.groups:
            oi = np.where(o)[0]
            Pm = np.linalg.inv(S[np.ix_(o, o)])
            R = self.Z[np.ix_(ix, oi)] - mu[o]
            g_mu = R @ Pm                                         # n_g x k
            G = 0.5 * (np.einsum("ij,ik->ijk", g_mu, g_mu) - Pm)  # n_g x k x k
            out[ix] = g_mu @ Dmu[oi] + np.einsum("ijk,jkp->ip", G, DS[np.ix_(oi, oi)])
        return out

    def robust_cov(self, h=1e-5):
        th = self.theta
        J = self.scores(th)
        Hm = np.zeros((self.n_par, self.n_par))
        for k in range(self.n_par):
            e = np.zeros(self.n_par); e[k] = h
            Hm[:, k] = (self.scores(th + e).sum(0) - self.scores(th - e).sum(0)) / (2 * h)
        Hm = 0.5 * (Hm + Hm.T)
        Hinv = np.linalg.inv(-Hm)
        return Hinv @ (J.T @ J) @ Hinv

    def std_paths(self):
        mu, S, Gx, b = self.unpack(self.theta)
        sd = np.sqrt(np.diag(S))
        a_std = [Gx[j, 0] * sd[0] / sd[Q + j] for j in range(6)]
        b_std = [b[j] * sd[Q + j] / sd[Q + 6] for j in range(6)]
        c_std = Gx[6, 0] * sd[0] / sd[Q + 6]
        return np.array(a_std), np.array(b_std), c_std


def chi_square(Z):
    """chi-square of the restricted model against the saturated one (same data), df = 15."""
    m1 = PathModel(Z, False).fit()
    m2 = PathModel(Z, True).fit()
    return 2 * (m1.fun - m2.fun), m1, m2
