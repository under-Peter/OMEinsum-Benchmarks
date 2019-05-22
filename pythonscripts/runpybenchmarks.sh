#!/bin/bash

source ./benchmarkpython/bin/activate
py.test --benchmark-storage="." --benchmark-save="benchnumpy" numpybenchmark.py
py.test --benchmark-storage="." --benchmark-save="benchtorch" torchbenchmark.py
deactivate

mv Linux-CPython-3.6-64bit/* ../benchmarkfiles/
trash Linux-CPython-3.6-64bit
