#!/bin/bash
####################################################
# setting up the environment for walking benchmark #
####################################################

set -e

# clone repos
git clone git@github.com:farzadab/walking-benchmark.git
git clone git@github.com:farzadab/roboschool.git
git clone git@github.com:belinghy/cassie_pybullet_env.git  # might be moved to UBCMOCCA soon

# create python virtualenv and 
# module load python/3.6
virtualenv venv
source venv/bin/activate
export PYTHON_INCLUDE=`pwd`/venv/include/python3.6m
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$PYTHON_INCLUDE

# build roboschool
# may need: sudo apt-get install qtbase5-dev libqt5opengl5-dev libassimp-dev cmake patchelf
# module load cmake qt/5.11.3 gcc/8.3.0
pushd roboschool
./install_boost.sh
./install_bullet.sh
source exports.sh
pushd roboschool/cpp-household
make clean
make -j4
make -j4 ../cpp_household.so
popd
pip install -e .
popd

cd walking-benchmark
ln -s ../roboschool/roboschool .
ln -s ../cassie_pybullet_env .