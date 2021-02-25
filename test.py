import os
import random
import numpy as np
import torch

from options.test_options import TestOptions
from data.dataloader import CreateDataLoader
from util.visualizer import save_images
from itertools import islice
from util import html, util
from models.ZUNIT import ZUNIT

def main():
    opt = TestOptions().parse()

    data_loader = CreateDataLoader(opt)

    model = ZUNIT()
    model.initialize(opt)

    web_dir = os.path.join(opt.results_dir, 'pics')
    webpage = html.HTML(web_dir, 'task {}'.format(opt.name))

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    for i, data in enumerate(islice(data_loader, opt.how_many)):
        print('process input image %3.3d/%3.3d' % (i, opt.how_many))
        all_images, all_names = model.translation(data)
        img_path = 'image%3.3i' % i
        for img, name in zip(all_images,all_names):
            save_images(webpage, img, name, img_path, None, width=opt.imageSize)
    webpage.save()

if __name__ == '__main__':
    main()