#!/bin/bash
######################################################
# setting up the walking benchmark on Compute Canada #
######################################################

set -e

## clone repos
git clone git@github.com:farzadab/walking-benchmark.git
git clone git@github.com:farzadab/roboschool.git
git clone git@github.com:UBCMOCCA/mocca_envs.git  # might require authentication
git clone git@github.com:DeepX-inc/machina.git

## create python virtualenv and 
module load python/3.6
virtualenv venv
source venv/bin/activate

## install PyTorch
pip install numpy torch_cpu --no-index  # alternatively can install `pytorch_gpu`
## install Tensorflow: just used for Tensorboard
pip install tensorflow_cpu

## install PyBullet: try to use the latest version from https://pypi.org/project/pybullet/#files
wget https://files.pythonhosted.org/packages/78/d2/42d6658f3311e60729dd38a5824378ed0b6a44a90b7719cd591aef27493e/pybullet-2.4.9.tar.gz
tar -xf pybullet-2.4.9.tar.gz
pushd pybullet-2.4.9
# the problematic line: uses too many cpu thread
sed -i -- 's/2\*multiprocessing.cpu_count()/10/g' setup.py
python setup.py install
popd

## install Machina
pushd machina
sed -i -- "s/'torch/#'torch/g" setup.py
sed -i -- "s/gym==/gym>=/g" setup.py
pip install pandas
python setup.py install
popd

# install pygit2
export LIBGIT2=$VIRTUAL_ENV
wget https://github.com/libgit2/libgit2/archive/v0.28.1.tar.gz
tar xzf v0.28.1.tar.gz
pushd libgit2-0.28.1/
# module load cmake gcc
cmake . -DCMAKE_INSTALL_PREFIX=$LIBGIT2
make
make install
export LDFLAGS="-Wl,-rpath='$LIBGIT2/lib',--enable-new-dtags $LDFLAGS"
pip install pygit2
tee -a ~/.zshrc <<EOF
export LD_LIBRARY_PATH=$LIBGIT2/lib:\$LD_LIBRARY_PATH
EOF
popd


## install mocca_envs
pushd mocca_envs
pip install -e .
popd

## install roboschool (hacky)
pip install roboschool

cd walking-benchmark
ln -s ../roboschool/roboschool .
cp ../venv/lib/python3.6/site-packages/roboschool/cpp_household.so roboschool


sed -i -- 's/roboschool/#roboschool/g' requirements.txt
sed -i -- 's/tensorflow/#tensorflow/g' requirements.txt
sed -i -- 's/machina-rl/#machina-rl/g' requirements.txt
sed -i -- 's/pybullet/#pybullet/g' requirements.txt
sed -i -- 's/torch/#torch/g' requirements.txt
sed -i -- 's/ipdb/#ipdb/g' requirements.txt
sed -i -- 's/pygit2/#pygit2/g' requirements.txt
pip install -r requirements.txt