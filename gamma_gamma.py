from defs import *
from scipy.stats import gamma
from scipy.special import gammaln, digamma, gammainc
from adam import Adam
from transform import *

# baseline model
# p(w) = gamma(theta, 1)
# q(w) = gamma(a, b)
class GammaGamma(object):
    def __init__(self, N, lr=0.1, theta=None):
        self.N = N
        self.optim = Adam(lr=lr)

        self.sp = Softplus()
        self.var = 0.01*np.random.normal(size=2*N+1)
        self.a = self.sp(self.var[0:N])
        self.b = self.sp(self.var[N:2*N])

        self.theta = np.random.gamma(1.,1.) if theta is None else theta
        self.var[-1] = self.sp.inv(self.theta)

    def reparam(self, debug=False):
        a = self.a
        b = self.b

        z = 0.2*np.ones(self.N) if debug else \
                1e-15 + np.random.rand(self.N)*(1.-1e-15)
        w = np.zeros(self.N)
        dwda = np.zeros(self.N)

        small = a < 1000
        if np.any(small):
            a_ = a[small]
            b_ = b[small]
            z_ = z[small]
            # reparam y ~ gamma(a+1, b)
            y_ = gamma.ppf(z_, a_+1, scale=b_**-1)
            dyda_ = (gamma.ppf(z_, a_+1+1e-5, scale=b_**-1)-y_)/1e-5

            u_ = 0.3*np.ones(a_.shape) if debug else \
                    1e-15 + np.random.rand(np.prod(a_.shape)).reshape(a_.shape)*(1.-1e-15)
            ua_ = u_**(1./a_)
            w[small] = ua_*y_
            dwda[small] = -log(u_)*w[small]/(a_**2) + ua_*dyda_

        large = np.logical_not(small)
        if np.any(large):
            a_ = a[large]
            b_ = b[large]
            sqa_ = np.sqrt(a_)
            z_ = 0.3*np.ones(a_.shape) if debug else np.random.normal(size=a_.shape)
            w[large] = (a_ + sqa_*z_)/b_
            dwda[large] = (1.+0.5*z_/sqa_)/b_
        dwdb = -w/b
        w[w<1e-40] = 1e-40

        self.w = w
        self.dwda = dwda
        self.dwdb = dwdb
        return w

    def sample_p(self):
        return np.random.gamma(self.theta, 1., self.N)

    def log_p(self, w):
        theta = self.theta
        lp = (theta-1)*log(w) - w - gammaln(theta)
        return lp.sum(lp.ndim-1).mean()

    def log_q(self, w):
        a = self.a
        b = self.b
        lq = a*log(b) + (a-1)*log(w) - b*w - gammaln(a)
        return lq.sum(lq.ndim-1).mean()

    def sample_q(self, S=1):
        return np.random.gamma(self.a, scale=self.b**-1, size=(S,self.N))

    def step(self, dlldw):
        w = self.w
        a = self.a
        b = self.b
        dwda = self.dwda
        dwdb = self.dwdb
        N = self.N
        theta = self.theta

        dlpdw = (theta-1)/(w+eps) - 1.
        dljdw = dlldw + dlpdw

        dlqda = log(b) + (a-1)*dwda/w + log(w) - b*dwda - digamma(a)
        dlqdb = a/b + (a-1)*dwdb/w - w - b*dwdb

        dLda = dljdw*dwda - dlqda
        dLdb = dljdw*dwdb - dlqdb
        dLdtheta = -N*digamma(theta) + log(w).sum()

        grad = np.append(np.concatenate(
                [dLda*self.sp.jacobian(a), dLdb*self.sp.jacobian(b)]),
                dLdtheta*self.sp.jacobian(theta))
        self.var = self.optim.step(self.var, -grad)

        self.a = self.sp(self.var[0:N])
        self.b = self.sp(self.var[N:2*N])
        self.theta = self.sp(self.var[-1])

    def get_hp_name(self):
        return ['theta']

    def get_hp(self):
        return [self.theta]

    def print_hp(self):
        return 'theta %.4f' % self.theta
