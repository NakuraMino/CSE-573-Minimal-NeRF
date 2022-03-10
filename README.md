# CSEP573-NeRF

<img src="./media/epoch=569-360v2.gif" width="300">

This is a semi-faithful pytorch-lightning implementation of Neural Radiance Fields (NeRF) from [NeRF: Representing Scenes as Neural Radiance for View Synthesis](https://arxiv.org/abs/2003.08934). We implement the whole repository from scratch with minimal guidance from the [original repository](https://github.com/bmild/nerf). In fact, we only borrow minimal code to generate rays and 360 degree views. We built the remaining training pipeline and models from scratch.

## Installation

We have a minimal `requirements.txt` file which can be installed using 

```
pip install -r requirements.txt
```

To clone this repository and download the lego dataset, run 

```
git clone https://github.com/NakuraMino/CSEP573-NeRF.git
cd CSEP573-NeRF/
./download_data.sh
```

## Training a Model 

To train a NeRF model, you should run 

```
!python train_nerf.py -n lego --gpu -s 120000 -rd ROOT_DIR \
                      -r 4096 full -b BASE_DIR -cr 0
```

where the `ROOT_DIR` is the root directory to save your model checkpoints, and `BASE_DIR` is the path to your data directory (should look like `./data/nerf_synthetic/lego/` or something along those lines).


## Viewing Results

So our repository is much simpler and bare bones than the original NeRF repository because *reasons* (but really, it's because I'm working by myself and this is a class project with a few weeks to work on this). Still, you can render a 360 view of your model:

```
python render.py -c CKPT_PATH -r 4096 -s SAVE_DIR
```

where `CKPT_PATH` is the path to the saved checkpoint model. It's probably in your `ROOT_DIR` from earlier. Lastly, `SAVE_DIR` is simply the place to save your `gif` that this script produces. Have fun!

## Repository Results

~Yes, my results are sad in comparison. Yes, I am very bummed. No, I have not found my bug/issue.~

We fixed it! This works now.

COMING SOON!!!

Here are some failures:

<img src="./media/epoch=569-360v2.gif" width="400">

## Acknowledgements / Citations

This work refers to code from the following open-source projects and datasets. I only borrowed code
from bmild's original NeRF repository, but I took a look at kwea123's and yenchinlin's codebases when
I got super confused. I don't think I borrowed code from either repositories but I'll put them in since
I referenced them a lot. Thanks :)

#### Neural Radiance Fields (bmild)
Original: https://github.com/bmild/nerf

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
Original: https://github.com/kwea123/nerf_pl

```
@misc{queianchen_nerf,
  author={Quei-An, Chen},
  title={Nerf_pl: a pytorch-lightning implementation of NeRF},
  url={https://github.com/kwea123/nerf_pl/},
  year={2020},
}
```

#### nerf-pytorch (yenchinlin)
Original: https://github.com/yenchenlin/nerf-pytorch

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