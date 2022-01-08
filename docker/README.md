## Create a simple docker image

To build the image run from the rc_rl root directory:

```
docker build -f docker/Dockerfile -t rc_rl .   
```

## Docker Images for Dopamine and EfficientZero

This image is extremely heavy since it contains a lot of dependencies and really unstable
since efficientZero and dopamine require different versions of python and some libraries.

It's ideal to have a separate image for each model.

```
docker build -f docker/dopamine/Dockerfile -t rc_rl_dopamine .   
```

```
docker build -f docker/efficientZero/Dockerfile -t rc_rl_efficient_zero .   
```

## Current images

Currently the images are hosted on dockerhub.

- https://hub.docker.com/r/sergioarnaud/rc_rl
- https://hub.docker.com/r/sergioarnaud/rc_rl_efficient_zero
- https://hub.docker.com/r/sergioarnaud/rc_rl_dopamine

## Running in different architectures

[Satori cluster](https://mit-satori.github.io) has a different architecture (`linux/ppc64le`), by default our docker image has the `linux/amd64`. To solve this we use the [docker buildx](https://github.com/docker/buildx) extension. 

```
docker buildx build --platform linux/amd64,linux/ppc64le -f docker/efficientZero/Dockerfile -t sergioarnaud/rc_rl_efficient_zero --push . 
```

> Docker buildx is installed by default in docker desktop, if you're on a server you have to install it [from source](https://github.com/docker/buildx#binary-release) before. 

> An issue with the architecture might arise ( `/bin/sh: Invalid ELF image for this architecture`) it can be solved by installing the right images using ``docker run --privileged --rm tonistiigi/binfmt --install all`

## Using singularity containers

Building a singularity image is straightforward when the docker image is used:

```
singularity pull rc_rl.sif docker://sergioarnaud/rc_rl
```
