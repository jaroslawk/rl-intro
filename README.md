
#  Introduction to reinforcement learning

Build image by executing::

```
docker build ./docker-image -t rl-intro-image
docker run -p 8888:8888 -v $HOME/code/rl-intro:/home/jovyan/work  rl-intro-image:latest
```


Preparing conda:
```
conda install -c akode gym
```