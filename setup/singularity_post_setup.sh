#!/bin/bash
##############################################################################
# no need to run this separately, this is called from `singularity_setup.sh` #
##############################################################################
cd roboschool/roboschool
cp -r /code/roboschool/roboschool/*.so .
cp -r /code/roboschool/roboschool/.libs/ .

cd ../../walking-benchmark
ln -s ../roboschool/roboschool .
ln -s ../mocca_envs/mocca_envs .
