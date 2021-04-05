# CTtoPET


## Tensorboard
 visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

```bash 
tensorboard --logdir=runs --port=8008
```

## Training 
```bash 
CUDA_VISIBLE_DEVICES=0 python trainCTtoPET.py -rl 'sum'
CUDA_VISIBLE_DEVICES=0 python trainCTtoPET.py -rl 'mean'
```
GPU=8GB -> batch~2,3 

GPU=16GB -> batch~16 

## Testing on NIFTI files (3D)
```bash 
CUDA_VISIBLE_DEVICES=0 python predict.py
```

## License

## Docker (personal)
```bash 
docker run -it --rm --gpus all --shm-size=100G --user $(id -u):$(id -g) --cpuset-cpus=10-19 \
-v /rsrch1/ip/msalehjahromi/Codes/CTtoPET:/home/msalehjahromi/CTtoPET \
-v /rsrch1/ip/msalehjahromi/data:/Data \
--name CTtoPET1 nnunetmori:latest

docker run -it --rm --gpus all --shm-size=100G --user $(id -u):$(id -g) --cpuset-cpus=20-29 \
-v /rsrch1/ip/msalehjahromi/Codes/CTtoPET:/home/msalehjahromi/CTtoPET \
-v /rsrch1/ip/msalehjahromi/data:/Data \
--name CTtoPET2 nnunetmori:latest

docker run -it --rm --gpus all --shm-size=100G --user $(id -u):$(id -g) --cpuset-cpus=30-39 \
-v /rsrch1/ip/msalehjahromi/Codes/CTtoPET:/home/msalehjahromi/CTtoPET \
-v /rsrch1/ip/msalehjahromi/data:/Data \
--name CTtoPET3 nnunetmori:latest
```