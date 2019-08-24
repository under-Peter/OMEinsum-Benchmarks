#!/bin/bash

source ./benchmarkpython/bin/activate
py.test --benchmark-storage="." --benchmark-save="torch-gpu" torchgpubenchmark.py
deactivate
