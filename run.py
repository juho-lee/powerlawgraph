import pickle
from tbfry_rgamma import *
from gamma_gamma import *
from grg import *
from utils import *
import time
import argparse
import os

def sgvb(model, train, test, batch_size, n_steps, eval_freq, estim_elbo,
        logfile=None):
    res = {}
    if estim_elbo:
        res['elbo'] = []
    res['tll'] = []
    hp_name = model.get_hp_name()
    for hpn in hp_name:
        res[hpn] = []

    bgen = BatchGenerator(train, batch_size)
    start = time.time()
    for t in range(n_steps):
        batch = bgen.next_batch()
        w = model.reparam()
        dlldw = log_likel_grad(batch, w)
        model.step(dlldw)
        if (t+1)%eval_freq == 0:
            line = 'step %d/%d (%.3f secs), ' % (t+1, n_steps, time.time()-start)
            w = model.sample_q(S=25)
            if estim_elbo:
                elbo = 0.
                n_train = len(train)
                if n_train > 1e+8:
                    for i in xrange(0, 100, n_train):
                        batch = train[i:min(i+100, n_train)]
                        elbo += log_likel(batch, w)
                else:
                    elbo += log_likel(train, w)
                elbo += model.log_p(w) - model.log_q(w)
                line += 'elbo %.4f, ' % elbo
                res['elbo'] = np.append(res['elbo'], elbo)

            tll = 0.
            n_test = len(test)
            if n_test > 1e+8:
                for i in xrange(0, 100, n_test):
                    batch = test[i:min(i+100, n_test)]
                    tll += log_likel(batch, w)
            else:
                tll += log_likel(test, w)
            line += 'test ll %.4f, ' % tll
            res['tll'] = np.append(res['tll'], tll)
            hp_val = model.get_hp()
            for (hpn, hpv) in zip(hp_name, hp_val):
                res[hpn] = np.append(res[hpn], hpv)

            line += model.print_hp()
            print line
            if logfile is not None:
                logfile.write(line+'\n')
    return res

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data name, generate if not provided',
            type=str, default=None)
    parser.add_argument('--N', help='number of nodes to generate',
            type=int, default=1000)
    parser.add_argument('--alpha', help='power-law parameter alpha for generation',
            type=float, default=0.7)
    parser.add_argument('--beta', help='sparsity parameter beta for generation',
            type=float, default=1.0)
    parser.add_argument('--lr', help='learning rate',
            type=float, default=0.1)
    parser.add_argument('--model', help='model to test',
            type=str, default='tbfry_rgamma')
    parser.add_argument('--no_cv', help='do cross validation for beta if True',
            action='store_false', dest='do_cv')
    parser.add_argument('--n_steps', help='number of steps to run',
            type=int, default=20000)
    parser.add_argument('--eval_freq', help='evaluation frequency',
            type=int, default=100)
    parser.add_argument('--batch_size', help='mini-batch size',
            type=int, default=0)
    parser.add_argument('--seed', help='random seed for train/test split',
            type=int, default=42)
    parser.add_argument('--no_elbo', help='estimate elbo if True',
            action='store_false', dest='estim_elbo')
    parser.add_argument('--savedir', help='dir to save results',
            type=str, default=None)
    args = parser.parse_args()

    if args.data is None:
        print 'generating graph...'
        model_true = tBFRYrGamma(args.N, alpha=args.alpha, beta=args.beta)
        w_true = model_true.sample_p()
        graph = sample_graph(w_true)
    else:
        filename = DATA_ROOT + '/' + args.data + '.pickle'
        with open(filename, 'rb') as f:
            graph = pickle.load(f)
    pairs = get_pairs(graph)
    train, test = train_test_split(pairs, seed=args.seed)
    N = graph['N']
    print 'The graph has %d nodes and %d connections' % (N, len(graph['row']))

    savedir = SAVE_ROOT + '/'
    if args.savedir is None:
        if args.data is None:
            savedir += 'synthetic_' + args.model
        else:
            savedir += args.data + '_' + args.model
    else:
        savedir += args.savedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    logfile = open(savedir+'/train.log', 'w', 0)

    if args.model == 'tbfry_rgamma':
        if args.do_cv:
            print '-------------------cross validation on beta-----------------------'
            n_folds = 5
            beta_list = [0.6, 0.8, 1.0, 1.2, 1.4]
            folds = cross_validation_split(train, n_folds)
            best_tll = -np.inf
            best_beta = beta_list[0]
            for beta in beta_list:
                avg_tll = 0.
                for i, (sub_train, valid) in enumerate(folds):
                    model = tBFRYrGamma(N, beta=beta)
                    print 'beta %f, %dth fold' % (beta, i+1)
                    res = sgvb(model, sub_train, valid, N,
                            args.n_steps/10, args.eval_freq, False)
                    avg_tll += res['tll'].max()
                avg_tll /= n_folds
                print 'beta %f, avg tll %f' % (beta, avg_tll)
                print
                if avg_tll > best_tll:
                    best_tll = avg_tll
                    best_beta = beta
            print 'best beta: %f, best tll: %.4f' % (best_beta, best_tll)
            print '------------------------------------------------------------------\n'
            logfile.write('best beta: %f\n' % best_beta)
            model = tBFRYrGamma(N, beta=best_beta, lr=args.lr)
        else:
            print 'beta: 1.0'
            logfile.write('beta: 1.0\n')
            model = tBFRYrGamma(N, beta=1.0, lr=args.lr)

    elif args.model == 'gamma_gamma':
        model = GammaGamma(N, lr=args.lr)
    else:
        raise NotImplementedError

    batch_size = N if args.batch_size is 0 else args.batch_size
    res = sgvb(model, train, test, batch_size,
            args.n_steps, args.eval_freq, args.estim_elbo, logfile=logfile)
    logfile.close()
    with open(savedir+'/results.pickle', 'wb') as f:
        pickle.dump(res, f)
