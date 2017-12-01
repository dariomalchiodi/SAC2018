
# coding: utf-8

# ## Data loading

# In[1]:

# Data format:
#
# n0, n1, ...
# mu0, k00, k01, ..., k0n
# ...
# mun, kn0, kn1, ..., knn

import csv
import gc
import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab as plt

from possibilearn import *
from possibilearn.kernel import PrecomputedKernel
from possibilearn.fuzzifiers import *

data_file_name = 'data/data-tettamanzi-complete.csv'

with open(data_file_name) as data_file:
    data = np.array(list(csv.reader(data_file)))

n = len(data) - 1

print '%d data items' % n

# ## Extract data names, membership values and Gram matrix

n = 1444

names = np.array(data[0])[1:n+1]
mu = np.array([float(row[0]) for row in data[1:n+1]])
gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n+1]]
                 for row in data[1:n+1]])

assert(len(names.shape) == 1)
assert(len(mu.shape) == 1)
assert(len(gram.shape) == 2)
assert(names.shape[0] == gram.shape[0] == gram.shape[1] == mu.shape[0])

# ## Compute adjustement in case of ill-conditioned Gram matrix

eigvals = np.linalg.eigvals(gram)
assert(sum([abs(e.imag) for e in eigvals]) < 1e-4)
abs_neg_eigvals = [-l.real for l in eigvals if l < 0]
adjustment = max(abs_neg_eigvals) if abs_neg_eigvals else 0
if adjustment:
    print('non PSD matrix: diagonal adjusment of {0}'.format(adjustment))

assert(len(mu)==n)

#    ## Membership and possibility learning through repeated hold-out

def estimate_possibility(n, mu, k, g, cs, ks, num_holdouts, percentages,
                         fuzzifier, verbose=False):
    axiom_indices = range(n)
    assert(len(axiom_indices)==len(mu)==n)

    paired_axioms = [axiom_indices[i:i+2] for i in range(0, n, 2)]
    paired_labels = [mu[i:i+2] for i in range(0, n, 2)]

    metrics_membership_rmse = []
    metrics_membership_median = []
    metrics_membership_stdev = []

    metrics_possibility_rmse = []
    metrics_possibility_median = []
    metrics_possibility_stdev = []

    for h in range(num_holdouts):
        (paired_values_train,
         paired_values_validate,
         paired_values_test,
         paired_mu_train,
         paired_mu_validate,
         paired_mu_test) = split_indices(paired_axioms, paired_labels,
                                         percentages)

        if verbose:
            print 'holdout {} of {}'.format(h, num_holdouts)

        best_c, _, result = model_selection_holdout(paired_values_train,
                                                    paired_mu_train,
                                                    paired_values_validate,
                                                    paired_mu_validate,
                                                    cs, ks,
                                                    sample_generator=g,
                                                    log=False,
                                                    adjustment=adjustment,
                                                    fuzzifier=fuzzifier,
                                                    verbose=verbose)
        if best_c is None:
            if verbose:
                print 'in holdout {} optimization always failed!'.format(h)
            continue

        if verbose:
            print 'in holdout {} best C is {}'.format(h, best_c)
        estimated_membership = result[0]

        # values and labels are still paired, we need to flatten them out
        values_test = flatten(paired_values_test)
        mu_test = flatten(paired_mu_test)

        membership_square_err = [(estimated_membership(v) - m)**2
                                 for v, m in zip(values_test, mu_test)]
        membership_rmse = math.sqrt(sum(membership_square_err) / len(values_test))
        metrics_membership_rmse.append(membership_rmse)

        membership_median = np.median(membership_square_err)
        metrics_membership_median.append(membership_median)

        membership_stdev = np.std(membership_square_err)
        metrics_membership_stdev.append(membership_stdev)

        estimated_mu = map(estimated_membership, values_test)
        actual_possibility = [mfi - mnotfi
                              for mfi, mnotfi in zip(mu_test[::2], mu_test[1::2])]
        estimated_possibility = [mfi - mnotfi
                                 for mfi, mnotfi in zip(estimated_mu[::2], estimated_mu[1::2])]

        possibility_square_err = [(actual - estimated)**2
                              for actual, estimated in zip(actual_possibility, estimated_possibility)]
        possibility_rmse = math.sqrt(sum(possibility_square_err) / len(possibility_square_err))
        metrics_possibility_rmse.append(possibility_rmse)

        possibility_median = np.median(possibility_square_err)
        metrics_possibility_median.append(possibility_median)

        possibility_stdev = np.std(possibility_square_err)
        metrics_possibility_stdev.append(possibility_stdev)

        indices = ['-'.join(map(str, pair)) for pair in paired_values_test]

        results = [(i, phi, notphi, max(phi, notphi), ephi, enotphi, max(ephi, enotphi), p, ep, (p - ep)**2)
                   for i, phi, notphi, p, ephi, enotphi, ep in zip(indices, mu_test[::2], mu_test[1::2], actual_possibility,
                                                            estimated_mu[::2], estimated_mu[1::2], estimated_possibility)]

        results.sort(key = lambda r: r[-1])

        with open('data/axioms-results-holdout-{}-{}-details.csv'.format(fuzzifier.name, h), 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(results)

        with open('data/axioms-results-holdout-{}-{}-global.csv'.format(fuzzifier.name, h), 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerows([
                ('membership RMSE', membership_rmse),
                ('membership median', membership_median),
                ('membership STDEV', membership_stdev),
                ('possibility RMSE', possibility_rmse),
                ('possibility median', possibility_median),
                ('possibility STDEV', possibility_stdev),
            ])

        errors = [r[-1] for r in results]
        p = plt.boxplot(errors)
        plt.savefig('data/axioms-results-holdout-{}-{}-boxplot.png'.format(fuzzifier.name, h))
        plt.clf()

        p = plt.hist(errors, bins=50)
        plt.savefig('data/axioms-results-holdout-{}-{}-histogram.png'.format(fuzzifier.name, h))
        plt.clf()

        gc.collect()

    if verbose:
        print 'Membership average values:'
        print 'RMSE: {}'.format(np.average(metrics_membership_rmse))
        print 'Median: {}'.format(np.average(metrics_membership_median))
        print 'STDEV: {}'.format(np.average(metrics_membership_stdev))

        print 'Possibility average values:'
        print 'RMSE: {}'.format(np.average(metrics_possibility_rmse))
        print 'Median: {}'.format(np.average(metrics_possibility_median))
        print 'STDEV: {}'.format(np.average(metrics_possibility_stdev))

    with open('data/axioms-results-holdout-{}-average-metrics.csv'.format(fuzzifier.name), 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerows([
            ('membership average RMSE', np.average(metrics_membership_rmse)),
            ('membership average median', np.average(metrics_membership_median)),
            ('membership average STDEV', np.average(metrics_membership_stdev)),
            ('possibility average RMSE', np.average(metrics_possibility_rmse)),
            ('possibility average median', np.average(metrics_possibility_median)),
            ('possibility average STDEV', np.average(metrics_possibility_stdev)),
        ])


# In[7]:

k = PrecomputedKernel(gram)
axiom_indices = range(n)

def g(m):
    return np.random.choice(axiom_indices, m if m <= len(axiom_indices) else len(axiom_indices))

#cs = (0.0062, 0.00625, 0.0063, 0.00635, 0.0064)
cs = (0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1, 10, 100)
ks = (k,)

num_holdouts = 10
percentages = (.8, .1, .1)

# In[6]:

fuzzifiers = [CrispFuzzifier(),
              LinearFuzzifier(),
              QuantileConstantPiecewiseFuzzifier(),
              QuantileLinearPiecewiseFuzzifier()] + \
             [ExponentialFuzzifier(alpha)
              for alpha in (.001, .005, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5)]

# In[8]:

for f in fuzzifiers:
    print 'starting experiments for fuzzifier {}'.format(f.latex_name)
    estimate_possibility(n, mu, k, g, cs, ks, num_holdouts, percentages, f, verbose=True)
    print 'experiments for fuzzifier {} ended'.format(f.latex_name)
