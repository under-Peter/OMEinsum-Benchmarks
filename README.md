# OMEinsum-Benchmarks

This repository is used to compare the performance of `einsum` in python (`numpy`, `torch`), with implementations in `julia`.

In `/pythonscripts`, the benchmarking-scripts for `numpy.einsum` and `torch.einsum` can be found.
The tested operations are:

| label | description | `einsum`-equation |
| --- | --- |---|
| `matmul`| matrix multiplication | `ij,jk -> ik`
| `dot`| dot product | `ijl,ijl->`
| `ptrace`| partial trace | `iij->j`
| `hadamard`| the hadamard product | `ijk,ijk->ijk`
| `tcontract`| tensor contraction | `ikl,kjl->ij`
| `indexsum`| summing over a (single) index | `ijk->ik`
| `trace`| (pairwise) tracing all indices | `ii->`
| `diag`| taking the diagonal w.r.t to two or more indices | `jii->ji`
| `starandcontract`| mixing tensor and star-contraction | `ij,il,il->j`
| `star`| starcontracting over a single index | `ij,ik,il->jkl`
| `outer`| taking the outer product between two tensors | `ij,kl->ijkl`
| `batchmul`| batchmultiplying two batches of matrices | `ijl,jkl->ikl`
| `perm`| permuting a tensors indices | `ijkl->ljki`

The benchmarks have been evaluated over inputs of different size,
labelled (ad-hoc) as `tiny`, `small`, `medium`, `large` and `huge`.
The current benchmark has been run with `torch`, `numpy` for python and
`OMEinsum.jl` and `Einsum.jl` for julia.

# Overview
The plots are split by the memory footprint of the operations,
i.e. by size size of the tensors involved which have been classified as tiny, small, medium, large or huge
rather arbitrarily.
The results below are for tensors of type `float64` but
more results can be found in the `plots/` folder.
Generally, changing to `float32` or complex types does not change
the picture too much, except that torch does not support complex numbers.

## tiny tensors
![](plots/float64-tiny.png)

## small tensors
![](plots/float64-small.png)

## medium tensors
![](plots/float64-medium.png)

## large tensors
![](plots/float64-large.png)

## huge tensors
![](plots/float64-huge.png)
