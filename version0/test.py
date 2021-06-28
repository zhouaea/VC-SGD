import mxnet as mx
import numpy as np

x = mx.nd.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(mx.nd.topk(x, axis=None, k=3, ret_typ='both'))