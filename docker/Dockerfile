FROM ubuntu:19.04

# get dependencies
RUN apt-get -y update && apt-get -y install python3-dev python3-pip curl \
               apt-utils libgl1-mesa-dev qtbase5-dev \
               libqt5opengl5-dev libassimp-dev patchelf cmake pkg-config git \
               wget vim htop

RUN bash -c "ln -s $(which python3) /usr/bin/python && ln -s $(which pip3) /usr/bin/pip"

RUN pip install tensorflow
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl

# install pybullet
RUN wget https://files.pythonhosted.org/packages/78/d2/42d6658f3311e60729dd38a5824378ed0b6a44a90b7719cd591aef27493e/pybullet-2.4.9.tar.gz && \
    tar -xf pybullet-2.4.9.tar.gz && \
    rm pybullet-2.4.9.tar.gz && \
    cd pybullet-2.4.9 && \
    sed -i -- 's/2\*multiprocessing.cpu_count()/10/g' setup.py && \
    python setup.py install && \
    cd .. && \
    rm -r pybullet-2.4.9

ENV CPATH /code

RUN mkdir -p $CPATH
WORKDIR $CPATH

# github password for cloning MOCCA envs
ARG pass

RUN git clone https://github.com/farzadab/walking-benchmark.git
RUN git clone https://github.com/farzadab/roboschool.git
RUN git clone "https://farzadab:$pass@github.com/UBCMOCCA/mocca_envs.git"

# set python and pip to point to python3 versions

# compile roboschool
RUN cd roboschool && \
     bash -c ". ./exports.sh && ./install_boost.sh && ./install_bullet.sh && ./roboschool_compile_and_graft.sh"     

# install pip package and test
RUN pip install -e roboschool
RUN python -c "import gym; gym.make('roboschool:RoboschoolAnt-v1').reset()"

# install mocca_envs and test
RUN pip install -e mocca_envs
# RUN python -c "import gym; gym.make('roboschool:RoboschoolAnt-v1').reset()"

WORKDIR $CPATH/walking-benchmark

# install other requirements
RUN sed -i -- 's/roboschool/#roboschool/g' requirements.txt && \
    sed -i -- 's/tensorflow/#tensorflow/g' requirements.txt && \
    sed -i -- 's/pybullet/#pybullet/g' requirements.txt && \
    sed -i -- 's/torch/#torch/g' requirements.txt && \
    sed -i -- 's/ipdb/#ipdb/g' requirements.txt && \
    sed -r -i -- 's/(.*)MOCCA/#\1MOCCA/g' requirements.txt && \
    pip install -r requirements.txt
