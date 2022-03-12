# CSEP573-NeRF
<p>
<img src="./media/model=lego-epoch=1089-360.gif" width="300">
<img src="./media/model=ship-epoch=1079-360.gif" width="300">
</p>
This is a semi-faithful pytorch-lightning implementation of Neural Radiance Fields (NeRF) from [NeRF: Representing Scenes as Neural Radiance for View Synthesis](https://arxiv.org/abs/2003.08934). We implement the whole repository from scratch with minimal guidance from the [original repository](https://github.com/bmild/nerf). In fact, we only borrow minimal code to generate rays, 360 degree views, etc. We built the remaining training pipeline and models from scratch.


## Why Another NeRF Implementation? 

<img src="./media/meme.jpg" width="300">

I believe in learning by doing! There were a billion details I would have never understood from simply reading the paper, so I implemented it. Also, because, well, you know. *School*.

NOTE: I don't think performance is as good as the original repository. Probably because some of my methods aren't as mathematically correct as some of the original repository? Whatever. 

## Installation

Setting up this repository is really simple and easy. 

```
# clone repository
!git clone https://github.com/NakuraMino/CSEP573-NeRF.git
%cd CSEP573-NeRF/

# install libraries not on colab default
!pip install -r requirements.txt

# download dataset
!./download_synthetic_data.sh MODEL
```

where MODEL is the model you want to train/render. Options are one of `chair`, `drums`, `ficus`, `hotdog`, `lego`, `materials`, `mic`, and `ship`.

## Training a Model 

To train a NeRF model, you should run 

```
!python train_nerf.py -n lego --gpu -s 120000 -rd ROOT_DIR \
                      -r 4096 full -b BASE_DIR -cr 0
```

where the `ROOT_DIR` is the root directory to save your model checkpoints, and `BASE_DIR` is the path to your data directory (should look like `./data/nerf_synthetic/lego/` or something along those lines). 
Some models, like `mic` and `ficus` will probably benefit from `-cr 1000`. This is a flag that crops the images to only sample from the center for the first 1000 iterations. It helps because these models
have a lot of empty backgrounds, which lead to more failures during the start of training (apparently).


## Viewing Results

So our repository is much simpler and bare bones than the original NeRF repository because *reasons* (but really, it's because I'm working by myself and this is a class project with a few weeks to work on this). Still, you can render a 360 view of your model:

```
python render.py -c CKPT_PATH -r 4096 -s SAVE_DIR
```

where `CKPT_PATH` is the path to the saved checkpoint model. It's probably in your `ROOT_DIR` from earlier. Lastly, `SAVE_DIR` is simply the place to save your `gif` that this script produces. Have fun!

## Repository Results

~Yes, my results are sad in comparison. Yes, I am very bummed. No, I have not found my bug/issue.~

We fixed it! This works now. I have limited computational resources so please forgive the fact I only have a meager amount of gifs compared to the original repository.
<p>
<img src="./media/model=lego-epoch=1089-360.gif" width="300">
<img src="./media/model=ship-epoch=1079-360.gif" width="300">
</p>
## Acknowledgements / Citations

This work refers to code from the following open-source projects and datasets. I only borrowed code
from bmild's original NeRF repository, but I took a look at kwea123's and yenchinlin's codebases when
I got super confused. I don't think I borrowed code from either repositories but I'll put them in since
I referenced them a lot. Thanks :)

#### Neural Radiance Fields (bmild)
Link: https://github.com/bmild/nerf

```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

#### nerf_pl (kwea123)
Link: https://github.com/kwea123/nerf_pl

```
@misc{queianchen_nerf,
  author={Quei-An, Chen},
  title={Nerf_pl: a pytorch-lightning implementation of NeRF},
  url={https://github.com/kwea123/nerf_pl/},
  year={2020},
}
```

#### nerf-pytorch (yenchinlin)
Link: https://github.com/yenchenlin/nerf-pytorch

```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
