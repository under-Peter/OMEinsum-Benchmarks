# run with $ py.test --benchmark-save="test" pytestbm.py
# will save in STORAGE-PATH (.benchmarks/.../)
import torch
import pytest
import numpy as np



@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2),
    np.random.rand(10,10),
    np.random.rand(10^2,10^2),
    np.random.rand(10^3,10^3)
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_matmul(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(np.einsum, "ij,jk->ik", arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2,3),
    np.random.rand(10,10,3),
    np.random.rand(10^2,10^2,3),
    np.random.rand(10^3,10^3,3)
    ],
    ids = ["tiny", "small", "medium", "large"])
def test_batchmul(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(np.einsum, "ijl,jkl->ikl", arr,arr)

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
    benchmark(np.einsum, "ijl,ijl->", arr,arr)

@pytest.mark.parametrize("dt",
        [np.float32, np.float64, np.complex64, np.complex128],
        ids = ["Float32", "Float64", "Complex{Float32}", "Complex{Float64}"])
@pytest.mark.parametrize("arr",[
    np.random.rand(2,2),
    np.random.rand(10,10),
    np.random.rand(10^2,10^2),
    np.random.rand(10^3,10^3)
    ],
    ids =["tiny", "small", "medium", "large"])
def test_trace(benchmark, arr, dt):
    arr = arr.astype(dt)
    benchmark(np.einsum, "ii->", arr)

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
    benchmark(np.einsum, "iij->j", arr)

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
    benchmark(np.einsum, "jii->ji", arr)

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
    benchmark(np.einsum, "ijkl->ljki", arr)

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
    benchmark(np.einsum, "ikl,kjl->ij", arr,arr)

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
    benchmark(np.einsum, "ij,ik,il->jkl", arr,arr,arr)

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
    benchmark(np.einsum, "ij,il,il->j", arr,arr,arr)

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
    benchmark(np.einsum, "ijk->ik", arr)

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
    benchmark(np.einsum, "ijk,ijk->ijk", arr,arr)

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
    benchmark(np.einsum, "ij,kl->ijkl", arr,arr)

