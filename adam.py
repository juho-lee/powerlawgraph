import numpy as np

class Adam(object):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1.0e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # timestep
        self.t = 0
        # first moment
        self.m = None
        # second moment
        self.v = None

    def step(self, x, g_t, decay=None, clip=None):
        # update timestep
        self.t += 1

        # update biased moments
        m_t = (1 - self.beta1)*g_t if self.m is None \
                else self.beta1*self.m + (1 - self.beta1)*g_t
        self.m = m_t
        v_t = (1 - self.beta2)*g_t*g_t if self.v is None \
                else self.beta2*self.v + (1 - self.beta2)*g_t*g_t
        self.v = v_t

        # compute biased-corrected moments
        m_hat_t = m_t / (1 - pow(self.beta1, self.t))
        v_hat_t = v_t / (1 - pow(self.beta2, self.t))

        # update parameter
        dx = self.lr * m_hat_t / (np.sqrt(v_hat_t) + self.eps)
        if clip is None:
            x = x - dx
        else:
            x = x - np.clip(dx, -clip, clip)
        return x

    def decay(self, rate):
        self.lr *= rate

# comparing simple learning rates vs Adam on logistic regression
def test():
    # generate input data
    dim = 100
    N = 10000
    X = np.random.normal(scale=np.sqrt(20), size=[N, dim])
    w_true = 0.5*np.ones([dim, 1])
    Py = 1./(1 + np.exp(-np.dot(X, w_true)))
    u = np.random.rand(N, 1)
    y = np.zeros([N, 1])
    y[u < Py] = 1.
    y[u >= Py] = -1.

    # regularization param
    lam = 0.01

    # objective function (with l2 regularizer)
    def f(w):
        return np.log(1 + np.exp(-y*np.dot(X, w))).mean() \
                + 0.5*lam*np.square(w).sum()

    # stochastic gradient of batch
    def grad(w, bind):
        Xb = X[bind]
        yb = y[bind]
        expterm = np.exp(-yb*np.dot(Xb, w))
        grad = -((yb*expterm*Xb)/(1 + expterm)).mean(0).reshape(-1, 1)
        grad = grad + lam*w
        return grad

    w_init = np.random.normal(size=[dim, 1], scale=0.01)

    # simple learning rate
    def simple_lr(t):
        return 0.1*pow(t + 1., -0.5)

    w = w_init
    n_epochs = 20
    fw = np.zeros(n_epochs)
    batch_size = 100
    n_batches = N / batch_size
    ind = range(N)
    t = 0
    for i in range(n_epochs):
        np.random.shuffle(ind)
        for j in range(n_batches):
            bind = ind[j*batch_size:(j+1)*batch_size]
            lr = simple_lr(t)
            t += 1
            w = w - lr*grad(w, bind)
        fw[i] = f(w)

    # optimization by adam
    w = w_init
    adam = Adam()
    fw_adam = np.zeros(n_epochs)
    for i in range(n_epochs):
        np.random.shuffle(ind)
        for j in range(n_batches):
            bind = ind[j*batch_size:(j+1)*batch_size]
            w = adam.step(w, grad(w, bind))
        fw_adam[i] = f(w)

    import matplotlib.pyplot as plt
    plt.plot(fw, label='simple')
    plt.plot(fw_adam, 'r', label='adam')
    plt.legend()
    plt.show()

    print fw[-1], fw_adam[-1]

if __name__=='__main__':
    test()
