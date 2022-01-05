To build the image run from the rc_rl root directory:

```
docker build -f docker/Dockerfile -t rc_rl .   
```

This image is extremely heavy since it contains a lot of dependencies and really unstable
since efficientZero and dopamine require different versions of python and some libraries.

It's ideal to have a separate image for each model.

```
docker build -f docker/dopamine/Dockerfile -t rc_rl_dopamine .   
```

```
docker build -f docker/efficientZero/Dockerfile -t rc_rl_efficient_zero .   
```

Currently the images are hosted on dockerhub.

- https://hub.docker.com/r/sergioarnaud/rc_rl
- https://hub.docker.com/r/sergioarnaud/rc_rl_efficient_zero
- https://hub.docker.com/r/sergioarnaud/rc_rl_dopamine

Building a singularity image is straightforward when the docker image is used:

```
singularity pull rc_rl.sif docker://sergioarnaud/rc_rl
```
