# run with $ py.test --benchmark-save="test" pytestbm.py
# will save in STORAGE-PATH (.benchmarks/.../)
import torch
import numpy
import pytest

#raise NameError('Please manually chose a cuda device, default is 4')

def manyinds(a,b):
    c = torch.einsum("abcdefghijklmnop,flnqrcipstujvgamdwxyz->bcdeghkmnopqrstuvwxyz",a,b)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("dim",[2], ids = ["tiny"])
def test_manyinds(benchmark, dim,  dt):
    str = "abcdefghijklmnop,flnqrcipstujvgamdwxyz->bcdeghkmnopqrstuvwxyz"
    inputs = str.split('-')[0]
    len1 = len(set(inputs.split(',')[0]))
    len2 = len(set(inputs.split(',')[1]))
    arr1_dims = [dim]*len1
    arr1 = numpy.random.random_sample(arr1_dims)
    arr1 = torch.tensor(arr1, dtype=dt)
    arr2_dims = [dim]*len2
    arr2 = numpy.random.random_sample(arr2_dims)
    arr2 = torch.tensor(arr2, dtype=dt)
    ca1 = arr1.cuda(4)
    ca2 = arr2.cuda(4)
    benchmark(manyinds, ca1, ca2)

def matmul(a,b):
    c = torch.einsum("ij,jk->ik",a,b)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10),
    numpy.random.rand(100,100),
    numpy.random.rand(1000,1000)
    ],
    ids = ["small", "medium", "large"])
def test_matmul(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    benchmark(matmul, ca,cb)

def batchmul(a,b):
    torch.einsum("ijl,jkl -> ikl", a, b)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10,3),
    numpy.random.rand(100,100,3),
    numpy.random.rand(1000,1000,3)
    ],
    ids = ["small", "medium", "large"])
def test_batchmul(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    benchmark(batchmul, ca, cb)

def mydot(a,b):
    torch.einsum("ijl,ijl -> ", a, b)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10,10),
    numpy.random.rand(20,20,20),
    numpy.random.rand(50,50,50),
    ],
    ids = ["small", "medium", "large"])
def test_dot(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    benchmark(mydot, ca, cb)

def mytrace(a):
    torch.einsum("ii -> ", a)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10),
    numpy.random.rand(100,100),
    numpy.random.rand(1000,1000)
    ],
    ids =["small", "medium", "large"])
def test_trace(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    benchmark(mytrace, ca)

def ptrace(a):
    torch.einsum("iij->j", a)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10,10),
    numpy.random.rand(20,20,20),
    numpy.random.rand(50,50,50),
    ],
    ids = ["small", "medium", "large"])
def test_ptrace(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    benchmark(ptrace, ca)

def mydiag(a):
    torch.einsum("jii->ji", a)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10,10),
    numpy.random.rand(20,20,20),
    numpy.random.rand(50,50,50),
    ],
    ids = ["small", "medium", "large"])
def test_diag(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    benchmark(mydiag, ca)


def myperm(a):
    torch.einsum("ijkl ->ljki", a)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,2,2),
    numpy.random.rand(10,10,10,10),
    numpy.random.rand(30,30,30,30),
    ],
    ids = ["small", "medium", "large"])
def test_perm(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    benchmark(myperm, ca)

def tcontract(a,b):
    torch.einsum("ikl,kjl -> ij", a, b)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,2),
    numpy.random.rand(10,10,10),
    numpy.random.rand(30,30,30),
    ],
    ids = ["small", "medium", "large"])
def test_tcontract(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    benchmark(tcontract, ca,cb)

def mystar(a,b,c):
    torch.einsum("ij,ik,il->jkl",a,b,c)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10),
    numpy.random.rand(30,30),
    numpy.random.rand(100,100),
    ],
    ids = ["small", "medium", "large"])
def test_star(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    c = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    cc = c.cuda(4)
    benchmark(mystar, ca, cb, cc)

def mystarandcontract(a,b,c):
    torch.einsum("ij,il,il->j", a,b,c)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10),
    numpy.random.rand(30,30),
    numpy.random.rand(100,100),
    ],
    ids = ["small", "medium", "large"])
def test_starandcontract(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    c = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    cc = c.cuda(4)
    benchmark(mystarandcontract, ca, cb, cc)

def myindexsum(a):
    torch.einsum("ijk->ik", a)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10,10),
    numpy.random.rand(30,30,30),
    numpy.random.rand(100,100,100),
    ],
    ids = ["small", "medium", "large"])
def test_indexsum(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    benchmark(myindexsum, ca)

def myhadamard(a,b):
    torch.einsum("ijk,ijk -> ijk", a, b)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10,10),
    numpy.random.rand(30,30,30),
    numpy.random.rand(100,100,100),
    ],
    ids = ["small", "medium", "large"])
def test_hadamard(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    benchmark(myhadamard, ca,cb)

def myouter(a,b):
    torch.einsum("ij,kl -> ijkl", a, b)
    torch.cuda.synchronize()

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(10,10),
    numpy.random.rand(50,50),
    numpy.random.rand(100,100),
    ],
    ids = ["small", "medium", "large"])
def test_outer(benchmark, arr, dt):
    a = torch.tensor(arr, dtype=dt)
    b = torch.tensor(arr, dtype=dt)
    ca = a.cuda(4)
    cb = b.cuda(4)
    benchmark(myouter, ca,cb)
