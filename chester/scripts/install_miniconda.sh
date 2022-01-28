#!/usr/bin/env bash
# Check this file before using
CONDA_INSTALL_PATH="~/software/miniconda3"
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_INSTALL_PATH
if [ -d $CONDA_INSTALL_PATH/bin ]; then
    PATH=$PATH:$HOME/bin
fi
echo 'PATH='$CONDA_INSTALL_PATH'/bin:$PATH' >> ~/.bashrc
rm ./Miniconda3-latest-Linux-x86_64.sh