MODEL=$1
mkdir -p data
cd data

gdown https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=sharing --fuzzy
unzip nerf_synthetic.zip "nerf_synthetic/$MODEL/*"