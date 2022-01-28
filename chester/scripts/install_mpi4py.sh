#!/usr/bin/env bash
echo "Install mpi4py 3.0.0"
cd /tmp
wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz
tar -zxf mpi4py-3.0.0.tar.gz
cd mpi4py-3.0.0
python setup.py build --mpicc=/usr/local/bin/mpicc
python setup.py install --user