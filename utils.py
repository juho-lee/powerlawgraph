import numpy as np

def train_test_split(data, holdout_ratio=0.2, seed=None):
    n_train = int((1-holdout_ratio)*len(data))
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed()
    return data[0:n_train], data[n_train:]

def cross_validation_split(data, n_folds, seed=None):
    start = 0
    fold_size = len(data)/n_folds
    folds = [0]*n_folds
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed()
    for i in range(n_folds):
        end = len(data) if i==n_folds-1 else start+fold_size
        train = np.concatenate([data[0:start], data[end:]])
        valid = data[start:end]
        folds[i] = (train, valid)
        start += fold_size
    return folds

class BatchGenerator(object):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.n_batches = len(data)/batch_size
        self.bind = 0

    def reset(self):
        self.bind = 0
        np.random.shuffle(self.data)

    def next_batch(self):
        batch = self.data[self.bind*self.batch_size:(self.bind+1)*self.batch_size]
        self.bind += 1
        if self.bind == self.n_batches:
            self.reset()
        return batch

if __name__ == '__main__':
    data = range(17)
    folds = cross_validation_split(data, 8, seed=3)
    for train, valid in folds:
        print train, valid
