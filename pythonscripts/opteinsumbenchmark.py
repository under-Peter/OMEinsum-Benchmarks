# run with $ py.test --benchmark-save="test" pytestbm.py
# will save in STORAGE-PATH (.benchmarks/.../)
import pytest
import numpy as np
import opt_einsum as oe

# high index count, complicated operation
@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("dim",[2], ids = ["tiny"])
def test_manyinds(benchmark, dim,  dt):
    str = "abcdefghijklmnop,flnqrcipstujvgamdwxyz->bcdeghkmnopqrstuvwxyz"
    inputs = str.split('-')[0]
    len1 = len(set(inputs.split(',')[0]))
    len2 = len(set(inputs.split(',')[1]))
    arr1_dims = [dim]*len1
    arr1 = np.random.random_sample(arr1_dims)
    arr1 = arr1.astype(dt)
    arr2_dims = [dim]*len2
    arr2 = np.random.random_sample(arr2_dims)
    arr2 = arr2.astype(dt)
    benchmark(oe.contract, str, arr1,arr2)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2),
    np.random.rand(10,10),
    np.random.rand(100,100),
    np.random.rand(1000,1000)
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_matmul(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ij,jk->ik", arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2,3),
    np.random.rand(10,10,3),
    np.random.rand(100,100,3),
    np.random.rand(1000,1000,3)
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_batchmul(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ijl,jkl->ikl", arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2,2),
    np.random.rand(10,10,10),
    np.random.rand(20,20,20),
    np.random.rand(50,50,50),
    np.random.rand(100,100,100)
    ],
    ids = ["tiny", "small", "medium", "large", "huge"])
def test_dot(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ijl,ijl->", arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2),
    np.random.rand(10,10),
    np.random.rand(100,100),
    np.random.rand(1000,1000)
    ],
    ids =["tiny", "small", "medium", "large"])
def test_trace(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ii->", arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2,2),
    np.random.rand(10,10,10),
    np.random.rand(20,20,20),
    np.random.rand(50,50,50),
    np.random.rand(100,100,100)
    ],
    ids = ["tiny", "small", "medium", "large", "huge"])
def test_ptrace(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "iij->j", arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2,2),
    np.random.rand(10,10,10),
    np.random.rand(20,20,20),
    np.random.rand(50,50,50),
    np.random.rand(100,100,100)
    ],
    ids = ["tiny", "small", "medium", "large", "huge"])
def test_diag(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark((lambda x,y: oe.contract(x,y).copy()), "jii->ji", arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(1,1,1,1),
    np.random.rand(2,2,2,2),
    np.random.rand(10,10,10,10),
    np.random.rand(30,30,30,30),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_perm(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark((lambda x,y: oe.contract(x,y).copy()), "ijkl->ljki", arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(1,1,1),
    np.random.rand(2,2,2),
    np.random.rand(10,10,10),
    np.random.rand(30,30,30),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_tcontract(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ikl,kjl->ij", arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2),
    np.random.rand(10,10),
    np.random.rand(30,30),
    np.random.rand(100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_star(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ij,ik,il->jkl", arr,arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2),
    np.random.rand(10,10),
    np.random.rand(30,30),
    np.random.rand(100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_starandcontract(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ij,il,il->j", arr,arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2,2),
    np.random.rand(10,10,10),
    np.random.rand(30,30,30),
    np.random.rand(100,100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_indexsum(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ijk->ik", arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2,2),
    np.random.rand(10,10,10),
    np.random.rand(30,30,30),
    np.random.rand(100,100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_hadamard(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ijk,ijk->ijk", arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2),
    np.random.rand(10,10),
    np.random.rand(50,50),
    np.random.rand(100,100),
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_outer(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(oe.contract, "ij,kl->ijkl", arr,arr)
