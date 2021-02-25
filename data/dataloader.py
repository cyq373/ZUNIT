from torch.utils.data import Dataset, DataLoader
from .BirdDataset import BirdDataset

def CreateDataLoader(opt):
    dataset = BirdDataset(opt.dataroot,
                        split='train' if opt.isTrain else 'test',
                        image_size=opt.imageSize,
                        is_flip = opt.is_flip,
                        )
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batchSize,
                             shuffle=True,
                             drop_last=True,
                             num_workers=opt.nThreads)
    return data_loader

