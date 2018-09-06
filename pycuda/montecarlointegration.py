import pycuda.autoinit
# import pycuda.driver as drv
import pycuda.curandom as curand
import pycuda.reduction
from pycuda.curandom import XORWOWRandomNumberGenerator
import numpy as np
# from pycuda import gpuarray
import timeit

nvalues = 10 ** 7
krnl = pycuda.reduction.ReductionKernel(np.float32, neutral="0",
        reduce_expr="a+b", map_expr="x[i] * x[i] + y[i] * y[i] < 1. ? 1 : 0",
        arguments="float *x, float *y")

rng = XORWOWRandomNumberGenerator()


def get_pi_cuda():
    a_device = rng.gen_uniform((nvalues,), dtype=np.float32)
    b_device = rng.gen_uniform((nvalues,), dtype=np.float32)

    # a_device = curand.rand(nvalues, dtype=np.float32)
    # b_device = curand.rand(nvalues, dtype=np.float32)
    hitcount = krnl(a_device, b_device).get()

    pi_est = hitcount / nvalues * 4
    print(f'CUDA result: {pi_est}')

def get_pi_numpy():
    data = np.random.rand(nvalues, 2)
    inside = len(np.argwhere(np.linalg.norm(data, axis=1) < 1))

    pi_est = (float(inside) / nvalues * 4.)
    print(f'numpy result: {pi_est}')


print(f'CUDA version took {timeit.timeit(get_pi_cuda, number=10)} s')
print(f'numpy version took {timeit.timeit(get_pi_numpy, number=10)} s')
