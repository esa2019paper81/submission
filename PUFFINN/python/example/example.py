from _puffinn import *
import numpy
import time
import math

d = 100
n = 10000
k = 10
n_queries = 100
s = 1 << 24 # ~ 16mb


index = Index('angular', d, s, hash_function = 'simhash', hash_source = 'independent')

print('Creating %d points (d = %d) and %d queries' % (n, d, n_queries))
dataset = [[numpy.random.normal(0, 1) for _ in range(d)]  for _ in range(n)]
queries = [[numpy.random.normal(0, 1) for _ in range(d)]  for _ in range(n_queries)]

for v in dataset:
    for i in range(len(v)):
        v[i] /= numpy.linalg.norm(v)
        
for v in queries:
    for i in range(len(v)):
        v[i] /= numpy.linalg.norm(v)

print('Computing ground truth')
ground_truth = [sorted([numpy.dot(q, v) for v in dataset])[-9:] for q in queries]

print('Building index')
for v in dataset:
    index.insert(v)

t0 = time.time()
index.rebuild()
print("Building index took %.2f seconds." % (time.time() - t0) )


results = []
print('Searching the index')
t0 = time.time()
for query in queries:
    results.append(index.search(query, k, 0.9999, 'none'))

print("Search the index took %.2f seconds." % (time.time() - t0))

found = 0

for i, r in enumerate(results):
    found += len([j for j in r if numpy.dot(queries[i], dataset[j]) >= (ground_truth[i][0] - 1e-4)])

print('Average recall: %f' % (found / (k * n_queries)))

    







