from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import os
import pandas as pd
import random
import pickle
from util.util import tensor2im, save_image

def prepare_data(data):
    img_s, cls_id_s, attr_s, img_t, cls_id_t, attr_t  = data

    img_s = Variable(img_s).cuda()
    cls_id_s = Variable(cls_id_s).cuda()
    attr_s = Variable(attr_s).cuda()

    img_t = Variable(img_t).cuda()
    cls_id_t = Variable(cls_id_t).cuda()
    attr_t = Variable(attr_t).cuda()

    return [img_s, cls_id_s, attr_s, img_t, cls_id_t, attr_t]

def translation(model, data):
    img_s, cls_id_s, attr_s, img_t, cls_id_t, attr_t = prepare_data(data)

    interSample,crossSample,interName,crossName = [],[],[],[]
    interSample.append([tensor2im(img_s[0].data)])
    interName.append(['source'])

    c_real_test = model.gen_test.enc_content(img_s)
    xt = model.gen_test.decode(c_real_test, attr_t)

    Num = 5
    for i in range(Num):
        attr = attr_s * (1 - i/Num) + attr_t * i/Num
        img = model.gen_test.decode(c_real_test, attr)
        interSample[0].append(tensor2im(img[0].data))
        interName[0].append('crossSample_{}'.format(i))

    interSample[0].append(tensor2im(xt[0].data))
    interName[0].append('fake')
    interSample[0].append(tensor2im(img_t[0].data))
    interName[0].append('target')
    return [interSample], [interName]