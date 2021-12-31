To build the image run from the rc_rl root directory:

```
docker build -f docker/Dockerfile -t rc_rl .   
```

Currently the image is hosted on dockerhub.

- https://hub.docker.com/r/sergioarnaud/rc_rl

Building a singularity image is straightforward when the docker image is used:

```
singularity pull rc_rl.sif docker://sergioarnaud/rc_rl
```
