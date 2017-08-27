from defs import *
import scipy.sparse
from itertools import combinations

def sample_graph(w):
    N = len(w)
    u = w/np.sqrt(w.sum())
    r = np.outer(u, u)
    p = r/(r + 1.)
    X = (np.random.rand(N, N) < p).astype(int)
    X = np.triu(X, k=1)

    graph = {}
    graph['N'] = N
    graph['row'], graph['col'], _ = scipy.sparse.find(X)

    return graph

def get_pairs(graph):
    N = graph['N']
    row = graph['row']
    col = graph['col']

    #enumerate all edges
    pairs = np.array(list(combinations(range(N), 2)))
    pairs = np.column_stack((pairs, np.zeros(len(pairs), dtype=int)))
    # fill in edges
    for (r, c) in zip(row, col):
        k = r*(2*N-r-1)/2-r + c-1
        pairs[k, 2] = 1

    return pairs

def get_degree(graph):
    N = graph['N']
    row = graph['row']
    col = graph['col']

    degree = np.zeros(N, dtype=int)
    for (r, c) in zip(row, col):
        degree[r] += 1
        degree[c] += 1
    return degree

def log_likel(pairs, w):
    if w.ndim == 1:
        w = w.reshape(1,-1)
    [S,N] = w.shape
    u = w/np.sqrt(w.sum(axis=1).reshape(-1,1))
    r = np.einsum('ij,ik->ijk', u, u)
    ll = -log(1 + r[:,pairs[:,0],pairs[:,1]]).sum(axis=1)
    pos = pairs[:,2] == 1
    ll += log(r[:,pairs[pos,0],pairs[pos,1]]).sum(axis=1)
    return ll.mean()

def log_likel_grad(pairs, w):
    N = len(w)
    wsum = w.sum()

    row = pairs[:,0]
    col = pairs[:,1]
    denom = 1./(wsum + np.outer(w, w)[row, col] + eps)
    gw = -(denom.sum())*np.ones(N)

    gw -= np.bincount(row, weights=w[col]*denom, minlength=N)
    gw -= np.bincount(col, weights=w[row]*denom, minlength=N)

    iw = 1./(w + eps)
    pos = pairs[:,2] == 1
    gw += (len(pairs) - pos.sum())/wsum
    prow = row[pos]
    pcol = col[pos]
    gw += np.bincount(prow, weights=iw[prow], minlength=N)
    gw += np.bincount(pcol, weights=iw[pcol], minlength=N)
    gw *= float(0.5*N*(N-1))/float(len(pairs))

    return gw
