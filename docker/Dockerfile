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
RUN apt-get install python3.8-dev -y

# Make python3.8 the default python.
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get install python3-distutils -y

# Install pip.
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# Install Dopamine dependencies.
COPY requirements.txt /root/rc_rl/
RUN pip install --no-cache-dir -r /root/rc_rl/requirements.txt

# Install JAX for GPU, overriding requirements.txt.
#RUN pip install --no-cache-dir --upgrade https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.74+cuda111-cp38-none-manylinux2010_x86_64.whl
RUN pip install --no-cache-dir --upgrade torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

WORKDIR /root/rc_rl



