from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections
import pickle
from torch.optim import lr_scheduler

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, cvt_rgb=True):
    if len(image_tensor.shape) == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1 and cvt_rgb:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def tensor2vec(vector_tensor):
    numpy_vec = vector_tensor.data.cpu().numpy()
    if numpy_vec.ndim == 4:
        return numpy_vec[:, :, 0, 0]
    else:
        return numpy_vec


def pickle_load(file_name):
    data = None
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def interp_z(z0, z1, num_frames, interp_mode='linear'):
    zs = []
    if interp_mode == 'linear':
        for n in range(num_frames):
            ratio = n / float(num_frames - 1)
            z_t = (1 - ratio) * z0 + ratio * z1
            zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    if interp_mode == 'slerp':
        # st()
        z0_n = z0 / (np.linalg.norm(z0)+1e-10)
        z1_n = z1 / (np.linalg.norm(z1)+1e-10)
        omega = np.arccos(np.dot(z0_n, z1_n))
        sin_omega = np.sin(omega)
        if sin_omega < 1e-10 and sin_omega > -1e-10:
            zs = interp_z(z0, z1, num_frames, interp_mode='linear')
        else:
            for n in range(num_frames):
                ratio = n / float(num_frames - 1)
                z_t = np.sin((1 - ratio) * omega) / sin_omega * z0 + np.sin(ratio * omega) / sin_omega * z1
                zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    return zs


def save_image(image_numpy, image_path,type='JPEG'):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path, type, quality=100)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1)).repeat(1, in_feat.size()[1], 1, 1)
    return in_feat / (norm_factor+eps)


def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    return torch.mean(torch.sum(in0_norm*in1_norm, dim=1))
    
def get_scheduler(optimizer, opt, last_epoch):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay+1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    return scheduler
