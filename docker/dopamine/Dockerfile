# If you want to use a different version of CUDA, view the available
# images here: https://hub.docker.com/r/nvidia/cuda
# Note:
#   - Jax currently supports CUDA versions up to 11.3.
#   - Tensorflow required CUDA versions after 11.2.
ARG cuda_docker_tag="11.4.2-cudnn8-devel-ubuntu20.04"
FROM nvidia/cuda:${cuda_docker_tag}

RUN apt-get update
# tzdata is required below. To avoid hanging, install it first.
RUN DEBIAN_FRONTEND="noninteractive" apt-get install tzdata -y
RUN apt-get install git wget libgl1-mesa-glx -y

# Install python3.8.
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install python3.8 -y

# Make python3.8 the default python.
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get install python3-distutils -y

# Install pip.
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# Install VGDL dependencies.
COPY requirements_vgdl.txt /root/rc_rl/
RUN pip install --no-cache-dir -r /root/rc_rl/requirements_vgdl.txt

# Install Dopamine dependencies.
COPY requirements_dopamine.txt /root/rc_rl/
RUN pip install --no-cache-dir -r /root/rc_rl/requirements_dopamine.txt

# Install JAX for GPU, overriding requirements.txt.
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html 

###### ATARI ######

# Copy ROMs into the image.
RUN mkdir /root/roms
COPY ./Roms.rar /root/roms/

RUN apt-get install rar unzip -y
RUN rar x /root/roms/Roms.rar /root/roms/

# Install ROMs with ale-py.
RUN pip install atari_py ale-py
RUN unzip /root/roms/ROMS.zip -d /root/roms
RUN python -m atari_py.import_roms /root/roms
RUN ale-import-roms /root/roms/ROMS

####################

WORKDIR /root/rc_rl
