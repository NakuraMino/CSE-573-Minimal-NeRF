# Configs

Training can be unstable for different datasets. While the default parameters should work for
most objects, some apparently need a few initial iterations that crop out the background. The 
authors cite `ficus` and `mic` (https://github.com/bmild/nerf/issues/29), but I've only trained 
`lego` and `ship` so far. Here are my configs for them. If I exclude a config, then you can
assume I used the default values.

## Lego:
```
!python train_nerf.py -n lego_nerf --gpu -s 120000 -rd ROOT_DIR \
                      -r 4096 full -b ./data/nerf_synthetic/lego/ -cr 0
```

## Ship:
```
!python train_nerf.py -n ship_nerf --gpu -s 120000 -rd ROOT_DIR \
                      -r 4096 full -b ./data/nerf_synthetic/ship/ -cr 1000
```
I saw that training `ship` was unstable without the cropping iters.