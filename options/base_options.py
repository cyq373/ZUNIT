import argparse
import os
from util import util
import pickle
import datetime
import dateutil.tz

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', default='../birds', help='path to images')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--imageSize', type=int, default=128, help='scale images to this size')
        self.parser.add_argument('--attr_dim', type=int, default=312, help='dim of attribute')
        self.parser.add_argument('--seen_classes_num', type=int, default=150, help='#of seen classes')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        self.parser.add_argument('--name', type=str, default='birds', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--is_flip', type=int, default=1, help='if >0, flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# sthreads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8000,  help='visdom display port')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--time_dir', type=str, default='', help='the experiment time dir')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        
        args = vars(self.opt)
    
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k).encode('utf-8'), str(v).encode('utf-8')))
        print('-------------- End ----------------')

        if self.isTrain:
            if not self.opt.continue_train:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
                self.opt.expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, timestamp)
                self.opt.model_dir = os.path.join(self.opt.expr_dir,'model')
                util.mkdirs(self.opt.model_dir)
                pkl_file = os.path.join(self.opt.expr_dir, 'opt.pkl')
                pickle.dump(self.opt, open(pkl_file, 'wb'))

                # save to the disk
                file_name = os.path.join(self.opt.expr_dir, 'opt_train.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k).encode('utf-8'), str(v).encode('utf-8')))
                    opt_file.write('-------------- End ----------------\n')
            else:
                self.opt.expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.time_dir)
                self.opt.model_dir = os.path.join(self.opt.expr_dir,'model')
                file_name = os.path.join(self.opt.expr_dir, 'opt_train_continue.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k).encode('utf-8'), str(v).encode('utf-8')))
                    opt_file.write('-------------- End ----------------\n')
        else:
            self.opt.expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.time_dir)
            if self.opt.results_dir=='':
                self.opt.results_dir = os.path.join(self.opt.expr_dir,'results')
            results_dir = self.opt.results_dir
            util.mkdirs(results_dir)
            self.opt.model_dir = os.path.join(self.opt.expr_dir,'model')
        return self.opt
