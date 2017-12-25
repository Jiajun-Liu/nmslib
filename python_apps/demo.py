import nmslib
import numpy

import time


# create a random matrix to index
data = numpy.random.randn(1000000, 100).astype(numpy.float32)
# initialize a new index, using a HNSW index on Cosine Similarity

s = time.time()
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)

s1 = time.time()

#
#
# # query for the nearest neighbours of the first datapoint
# ids, distances = index.knnQuery(data[1:1000], k=10)

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
neighbours = index.knnQueryBatch(data[:50000], k=100, num_threads=16)

s2 = time.time()

print('time for indexing %d, search %d' % (s1-s,s2-s1))