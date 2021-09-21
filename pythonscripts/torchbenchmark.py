# run with $ py.test --benchmark-save="test" pytestbm.py
# will save in STORAGE-PATH (.benchmarks/.../)
import torch
import numpy
import pytest


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
    benchmark(torch.einsum, str, arr1,arr2)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2),
    numpy.random.rand(10,10),
    numpy.random.rand(100,100),
    numpy.random.rand(1000,1000)
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_matmul(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ij,jk->ik", arr,arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,3),
    numpy.random.rand(10,10,3),
    numpy.random.rand(100,100,3),
    numpy.random.rand(1000,1000,3)
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_batchmul(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ijl,jkl->ikl", arr,arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,2),
    numpy.random.rand(10,10,10),
    numpy.random.rand(20,20,20),
    numpy.random.rand(50,50,50),
    numpy.random.rand(100,100,100)
    ],
    ids = ["tiny", "small", "medium", "large", "huge"])
def test_dot(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ijl,ijl->", arr,arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2),
    numpy.random.rand(10,10),
    numpy.random.rand(100,100),
    numpy.random.rand(1000,1000)
    ],
    ids =["tiny", "small", "medium", "large"])
def test_trace(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ii->", arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,2),
    numpy.random.rand(10,10,10),
    numpy.random.rand(20,20,20),
    numpy.random.rand(50,50,50),
    numpy.random.rand(100,100,100)
    ],
    ids = ["tiny", "small", "medium", "large", "huge"])
def test_ptrace(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "iij->j", arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,2),
    numpy.random.rand(10,10,10),
    numpy.random.rand(20,20,20),
    numpy.random.rand(50,50,50),
    numpy.random.rand(100,100,100)
    ],
    ids = ["tiny", "small", "medium", "large", "huge"])
def test_diag(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "jii->ji", arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(1,1,1,1),
    numpy.random.rand(2,2,2,2),
    numpy.random.rand(10,10,10,10),
    numpy.random.rand(30,30,30,30),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_perm(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ijkl->ljki", arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(1,1,1),
    numpy.random.rand(2,2,2),
    numpy.random.rand(10,10,10),
    numpy.random.rand(30,30,30),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_tcontract(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ikl,kjl->ij", arr,arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2),
    numpy.random.rand(10,10),
    numpy.random.rand(30,30),
    numpy.random.rand(100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_star(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ij,ik,il->jkl", arr,arr,arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2),
    numpy.random.rand(10,10),
    numpy.random.rand(30,30),
    numpy.random.rand(100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_starandcontract(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ij,il,il->j", arr,arr,arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,2),
    numpy.random.rand(10,10,10),
    numpy.random.rand(30,30,30),
    numpy.random.rand(100,100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_indexsum(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ijk->ik", arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2,2),
    numpy.random.rand(10,10,10),
    numpy.random.rand(30,30,30),
    numpy.random.rand(100,100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_hadamard(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ijk,ijk->ijk", arr,arr)

@pytest.mark.parametrize("dt",
        [torch.float32, torch.float64],
        ids = ["Float32", "Float64"])
@pytest.mark.parametrize("arr",[
    numpy.random.rand(2,2),
    numpy.random.rand(10,10),
    numpy.random.rand(50,50),
    numpy.random.rand(100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_outer(benchmark, arr, dt):
    arr = torch.tensor(arr, dtype=dt)
    benchmark(torch.einsum, "ij,kl->ijkl", arr,arr)
