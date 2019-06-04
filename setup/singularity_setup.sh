#!bin/bash
########################################################################
# setting up the walking benchmark on Compute Canada using Singularity #
########################################################################

#### client-side to create the image and push it to Docker Hub ###################
# cd docker
# docker build --tag=benchmark . --build-arg pass="*****"  # TODO: password
# docker tag benchmark farzadab/walking-benchmark
# docker push farzadab/walking-benchmark
## or 
# sudo singularity build --writable imgwalking.sif singularity.recipe
##################################################################################

# load singularity and pull from Docker Hub
module load singularity/3.2
singularity pull imgwalking.sif docker://farzadab/walking-benchmark

# pulling Git repos: requires authentication
git clone git@github.com:farzadab/walking-benchmark.git
git clone git@github.com:farzadab/roboschool.git
git clone cassie_mocap git@github.com:UBCMOCCA/mocca_envs.git
git clone git@github.com:DeepX-inc/machina.git

# post setup (roboschool libraries)
singularity exec -B /home -B /project -B /scratch \
    imgwalking.sif walking-benchmark/setup/singularity_post_setup.sh


#### usage: interactive shell #######################################
# singularity shell -B /home -B /project -B /scratch imgwalking.sif
#####################################################################
