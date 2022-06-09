# ZUNIT

Pytorch implementation of our paper: ["Zero-shot unsupervised image-to-image translation via exploiting semantic attributes"](https://www.sciencedirect.com/science/article/pii/S0262885622001184).

<p align="center">
<img src='images/framework.png'  align="center" width='90%'>
</p>

### Dependencies
you can install all the dependencies  by
```
pip install -r requirements.txt
```

### Datasets
- Download [CUB](https://drive.google.com/file/d/1imQTOhiXlc1i9BnFQJD3XTFmroxCZlPQ/view?usp=sharing) dataset.
- Unzip the birds.zip at `./dataset`.

### Training
- To view training results and loss plots, run
```
python -m visdom.server -p 8080
```
and click the URL http://localhost:8080. 

- Run
```
bash ./scripts/train_bird.sh
```

### Testing
- Run
```
bash ./scripts/test_bird.sh
```
- The testing results will be saved in `checkpoints/{exp_name}/results` directory.

### Results
<p align="center">
<img src='images/results1.png'  align="center" width='90%'>
</p>

<p align="center">
<img src='images/results2.png'  align="center" width='90%'>
</p>

### Bibtex
If this work is useful for your research, please consider citing :
```
@article{CHEN2022104489,
title = {Zero-shot unsupervised image-to-image translation via exploiting semantic attributes},
journal = {Image and Vision Computing},
pages = {104489},
year = {2022},
issn = {0262-8856},
doi = {https://doi.org/10.1016/j.imavis.2022.104489},
author = {Yuanqi Chen and Xiaoming Yu and Shan Liu and Wei Gao and Ge Li}
}
```

### Acknowledgement
The code used in this research is inspired by [DMIT](https://github.com/Xiaoming-Yu/DMIT) and [FUNIT](https://github.com/NVlabs/FUNIT).

### Contact
Feel free to contact me if there is any questions (cyq373@pku.edu.cn).
