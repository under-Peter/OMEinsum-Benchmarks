#!/bin/bash

source ./benchmarkpython/bin/activate
py.test --benchmark-storage="." --benchmark-save="benchopteinsum" opteinsumbenchmark.py
py.test --benchmark-storage="." --benchmark-save="benchnumpy" numpybenchmark.py
py.test --benchmark-storage="." --benchmark-save="benchtorch" torchbenchmark.py
py.test --benchmark-storage="." --benchmark-save="torch-gpu" torchgpubenchmark.py
py.test --benchmark-storage="." --benchmark-save="opt_einsum-gpu" opteinsumgpubenchmark.py
deactivate

mv Linux-CPython-3.8-64bit/* ../benchmarkfiles/
rm -r Linux-CPython-3.8-64bit
