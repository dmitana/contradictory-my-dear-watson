# Contradictory, My Dear Watson

## Prerequisites 
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

Native GPU support has not landed in docker-compose yet. For now install patched versions of `docker-py` and `docker-compose` as mentioned [here](https://github.com/docker/compose/issues/6691#issuecomment-571309691):
```bash
pip install --user git+https://github.com/docker/docker-py.git
pip install --user git+https://github.com/yoanisgil/compose.git@device-requests
```

## Getting Started
1. Build docker image:
```bash
$ COMPOSE_API_VERSION=auto docker-compose up --build -d
```

2. Docker container is running after building. Next time, docker container can be started and stopped as follows:
```bash
# Start docker container
$ docker-compose start

# Stop docker container
$ docker-compose stop
```
