PATH=~/software/miniconda3/bin:~/anaconda3/bin:$PATH
cd softgym
. prepare_1.0.sh
cd ..
export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export EGL_GPU=$CUDA_VISIBLE_DEVICES
