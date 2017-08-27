from gamma_gamma import *
import warnings
from scipy.integrate import quad

# bfry model
# p(w) = tBFRY(alpha, C=N**beta)
# q(w) = rGamma(a, b, C=N**beta)
class tBFRYrGamma(GammaGamma):
    def __init__(self, N, lr=0.1, alpha=None, beta=1.0):
        self.N = N
        self.C = N**beta
        self.optim = Adam(lr=lr)

        alpha_ub = min(1.0, 1./beta)
        self.sp = Softplus()
        self.sig = Sigmoid(alpha_ub)

        self.var = 0.01*np.random.normal(size=2*N+1)
        self.a = self.sp(self.var[0:N])
        self.b = self.sp(self.var[N:2*N])

        self.alpha = np.random.beta(1.,1.) if alpha is None else alpha
        self.alpha = min(self.alpha, alpha_ub)
        self.var[-1] = self.sig.inv(self.alpha)

    def reparam(self, debug=False):
        super(tBFRYrGamma, self).reparam(debug=debug)
        over = self.w >= self.C
        self.w[over] = self.C
        self.dwda[over] = 0.
        self.dwdb[over] = 0.
        return self.w

    def sample_p(self):
        N = self.N
        C = self.C
        alpha = self.alpha
        w = np.random.gamma(1.-alpha, 1., N) / np.random.beta(alpha, 1., N)
        to_resample = w > C
        num_to_resample = to_resample.sum()
        tol = 0
        while num_to_resample > 0:
            if tol > 10:
                w[to_resample] = C
                break
            w[to_resample] = np.random.gamma(1.-alpha, 1., num_to_resample) / \
                    np.random.beta(alpha, 1., num_to_resample)
            to_resample = w > C
            num_to_resample = to_resample.sum()
            tol += 1
        return w

    def log_p(self, w):
        C = self.C
        alpha = self.alpha
        if np.any(np.logical_or(w>C, w<0)):
            return -np.inf
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = lambda x: (x**(-alpha-1))*(1-exp(-x))
            Z, _ = quad(f, 0.0, C)
        lp = (-alpha-1)*log(w) + log(1 - exp(-w)) - log(Z)
        return lp.sum(lp.ndim-1).mean()

    def sample_q(self, S=1):
        w = super(tBFRYrGamma, self).sample_q(S=S)
        w[w>=self.C] = self.C
        return w

    def log_q(self, w):
        a = np.tile(self.a, (w.shape[0],1)) if w.ndim==2 else self.a
        b = np.tile(self.b, (w.shape[0],1)) if w.ndim==2 else self.b
        C = self.C
        lq = a*log(b) + (a-1)*log(w) - b*w - gammaln(a)
        over = w >= C
        lq[over] = log(1.-gammainc(a[over], b[over]*C))
        return lq.sum(lq.ndim-1).mean()

    def step(self, dlldw):
        w = self.w
        a = self.a
        b = self.b
        dwda = self.dwda
        dwdb = self.dwdb
        N = self.N
        C = self.C
        alpha = self.alpha

        dlpdw = -(1+alpha)/(w+eps) + exp(-w - log(1-exp(-w)))
        dljdw = dlldw + dlpdw

        dlqda = log(b) + (a-1)*dwda/w + log(w) - b*dwda - digamma(a)
        dlqdb = a/b + (a-1)*dwdb/w - w - b*dwdb
        over = w >= C
        a_ = a[over]
        b_ = b[over]
        p = gammainc(a_, b_*C)
        dlqda[over] = (p-gammainc(a_+1e-5, b_*C))/1e-5/(1.-p+eps)
        dlqdb[over] = -exp(a_*log(C) + (a_-1)*log(b_) - b_*C - gammaln(a_))/(1.-p+eps)

        dLda = dljdw*dwda - dlqda
        dLdb = dljdw*dwdb - dlqdb

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = lambda x: (x**(-alpha-1))*(1-exp(-x))
            Z, _ = quad(f, 0.0, C)
            f = lambda x: -(x**(-alpha-1)*log(x)*(1-exp(-x)))
            gZ, _ = quad(f, 0.0, C)
        dLdalpha = -N*gZ/Z - log(w).sum()

        grad = np.append(np.concatenate(
            [dLda*self.sp.jacobian(a), dLdb*self.sp.jacobian(b)]),
            dLdalpha*self.sig.jacobian(alpha))

        self.var = self.optim.step(self.var, -grad)

        self.a = self.sp(self.var[0:N])
        self.b = self.sp(self.var[N:2*N])
        self.alpha = self.sig(self.var[-1])

    def get_hp_name(self):
        return ['alpha']

    def get_hp(self):
        return [self.alpha]

    def print_hp(self):
        return 'alpha %.4f' % self.alpha
