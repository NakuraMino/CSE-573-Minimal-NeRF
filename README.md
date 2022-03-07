# CSEP573-NeRF

This is a semi-faithful pytorch-lightning implementation of Neural Radiance Fields (NeRF) from [NeRF: Representing Scenes as Neural Radiance for View Synthesis](https://arxiv.org/abs/2003.08934). We implement the whole repository from scratch with minimal guidance from the [original repository](https://github.com/bmild/nerf). In fact, we only borrow minimal code to generate ray directions, origins, and 360 degree views and build the remaining training pipeline and models from scratch.

### Installation

I don't have a `requirements.txt` or `env.yml` file becuase of *reasons*, but you really only need to run `pip install wandb`, `pip install pytorch-lightning` and install `pytorch`.

To clone this repository and download the lego dataset, run 

```
git clone https://github.com/NakuraMino/CSEP573-NeRF.git
cd CSEP573-NeRF/
./download_data.sh
```

### Training a Model 

To train a NeRF model, you should run 

```
python train_nerf.py -n lego_nerf --gpu -s 500000 -rd ROOT_DIR \
                      -r 4096 full -b BASE_DIR -cr 1000
```

where the `ROOT_DIR` is the root directory to save your model checkpoints, and `BASE_DIR` is the path to your data directory (should look like `./data/nerf_synthetic/lego/` or something along those lines).


### Viewing Results

So our repository is much simpler and bare bones than the original NeRF repository because *reasons* (but really, it's because I'm working by myself and this is a class project with a few weeks to work on this). Still, you can render a 360 view of your model:

```
python render.py -c CKPT_PATH -r 4096 -s SAVE_DIR
```

where `CKPT_PATH` is the path to the saved checkpoint model. It's probably in your `ROOT_DIR` from earlier. Lastly, `SAVE_DIR` is simply the place to save your `gif` that this script produces. Have fun!

### Repository Results

Yes, my results are sad in comparison. Yes, I am very bummed. No, I have not found my bug/issue.

![lego-360.gif](./recons/epoch=1389-360.gif)