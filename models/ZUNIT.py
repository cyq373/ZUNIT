from __future__ import print_function
from collections import OrderedDict
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid
from util.util import tensor2im, get_scheduler
from util.translation import translation
from models.networks import Generator, Discriminator, weights_init

################## ZUNIT #############################
class ZUNIT():
    def name(self):
        return 'ZUNIT'

    def initialize(self, opt):
        torch.cuda.set_device(opt.gpu)
        cudnn.benchmark = True
        
        self.opt = opt
        self.build_models()

    def build_models(self):
        ################### encoder #########################################
        
        self.G = Generator(attr_dim=self.opt.attr_dim)
        ################### decoder ###########################################
        if self.opt.isTrain:
            ################### discriminators #####################################
            self.D = Discriminator(num_classes=self.opt.seen_classes_num, attr_dim=self.opt.attr_dim)

            ################### init_weights ########################################
            if self.opt.continue_train:
                self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.D.load_state_dict(torch.load('{}/D_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            else:
                self.G.apply(weights_init(self.opt.init_type))
                self.D.apply(weights_init(self.opt.init_type))
            self.G.cuda()
            self.D.cuda()
            ################## define optimizers #####################################
            self.define_optimizers()
            self.define_lr_schedulers()
        else:
            self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            self.G.cuda()
            self.G.eval()

    def define_optimizer(self, Net):
        return optim.Adam(Net.parameters(),
                                    lr=self.opt.lr,
                                    betas=(0.5, 0.999))

    def define_optimizers(self):
        self.G_opt = self.define_optimizer(self.G)
        self.D_opt = self.define_optimizer(self.D)
        if self.opt.continue_train:
            self.G_opt.load_state_dict(torch.load('{}/G_opt_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            self.D_opt.load_state_dict(torch.load('{}/D_opt_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))

    def define_lr_schedulers(self):
        last_epoch = int(self.opt.which_epoch) if self.opt.continue_train else -1
        self.lr_schedulers = []
        self.lr_schedulers.append(get_scheduler(self.G_opt,self.opt,last_epoch))
        self.lr_schedulers.append(get_scheduler(self.D_opt,self.opt,last_epoch))
        lr = self.G_opt.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def update_lr(self):
        for scheduler in self.lr_schedulers:
            scheduler.step()
        lr = self.G_opt.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save(self, name):
        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.opt.model_dir, name))

    def prepare_data(self, data):
        img, cls_id, attr, attr_u = data
        self.real = Variable(img).cuda()
        self.attr_source = Variable(attr).cuda()
        self.attr_unseen = Variable(attr_u).cuda()
        cls_id = Variable(cls_id).cuda()
        return self.real, self.attr_source, cls_id, self.attr_unseen

    def translation(self, data):
        return translation(self.G, data)

    def get_current_errors(self):
        dict = []
        dict += [('l_real', self.l_real.item())]
        dict += [('l_fake', self.l_fake.item())]
        dict += [('l_reg', self.l_reg.item())]
        dict += [('l_adv', self.l_adv.item())]
        dict += [('l_x_rec', self.l_x_rec.item())]
        dict += [('l_sem', self.l_sem.item())]
        dict += [('l_cls_real', self.l_cls_real.item())]
        dict += [('l_cls_xr', self.l_cls_xr.item())]
        dict += [('l_cls_xt', self.l_cls_xt.item())]
        dict += [('l_cls_rand', self.l_cls_rand.item())]

        return OrderedDict(dict)

    def get_current_visuals(self):
        with torch.no_grad():
            self.G.eval()
            self.G.gen.eval()
            self.G.gen_test.eval()

            rand_index = torch.cat([torch.randperm(self.real.size(0)//2)+self.real.size(0)//2,torch.randperm(self.real.size(0)//2)],dim=0) 
            attr_source = self.attr_source
            attr_target = attr_source[rand_index]

            c_real = self.G.gen_test.enc_content(self.real)
            xr, _ = self.G.gen_test.decode(c_real, attr_source)
            xt, _ = self.G.gen_test.decode(c_real, attr_target)
            x_unseen, _ = self.G.gen_test.decode(c_real, self.attr_unseen)

            self.G.train()
            
            real = make_grid(self.real.data,nrow=1,padding=0)
            rec = make_grid(xr.data,nrow=1,padding=0)
            fake_unseen = make_grid(x_unseen.data,nrow=1,padding=0)
            fake_target = make_grid(xt.data,nrow=1,padding=0)
            imgs = [real, rec, fake_unseen, fake_target]
            targetReal = make_grid(self.real[rand_index].data,nrow=1,padding=0)
            imgs.append(targetReal)
            imgs = torch.cat(imgs,2)
            return OrderedDict([('real, rec, fake_unseen, fake_target, target',tensor2im(imgs))])

    def update_model(self, data):
        ### prepare data ###
        real, attr_source, cls_id, attr_unseen = self.prepare_data(data)
        targetIndex = torch.cat([torch.randperm(real.size(0)//2)+real.size(0)//2,torch.randperm(real.size(0)//2)],dim=0) 
        attr_target = attr_source[targetIndex]

        ### generate image ###
        c_real = self.G.gen.enc_content(real)
        xr, emd_s = self.G.gen.decode(c_real, attr_source)
        xt, emd_t = self.G.gen.decode(c_real, attr_target)
        x_rand, emd_r = self.G.gen.decode(c_real, attr_unseen)

        ### update dis ###
        self.D.zero_grad()

        real.requires_grad_()
        self.l_real, self.acc_r, resp_r, self.l_cls_real = self.D.calc_dis_real_loss(real, cls_id, attr_source)
        l_reg_pre = self.D.calc_grad2(resp_r, real)
        self.l_reg = 10 * l_reg_pre

        self.l_fake, self.acc_f, resp_f, _ = self.D.calc_dis_fake_loss(xt.detach(), cls_id[targetIndex], attr_target)

        l_dis_total = self.l_fake + self.l_real + self.l_reg + self.l_cls_real
        l_dis_total.backward()

        self.D_opt.step()

        ### update gen ###
        self.G.zero_grad()

        l_adv_t, gacc_t, xt_gan_feat, self.l_cls_xt = self.D.calc_gen_loss(xt, cls_id[targetIndex], attr_target)
        l_adv_r, gacc_r, xr_gan_feat, self.l_cls_xr = self.D.calc_gen_loss(xr, cls_id, attr_source)
        self.l_cls_rand = self.D.calc_unseen_loss(x_rand, attr_unseen)
        self.l_x_rec = torch.mean(torch.abs(xr - real)) * 0.1
        self.l_adv = 0.5 * (l_adv_t + l_adv_r)

        # cosine distance
        cos = torch.nn.CosineSimilarity(dim=1)
        self.l_sem = torch.nn.MSELoss()(cos(attr_source, attr_target), cos(emd_s, emd_t)) * 10

        l_gen_total = self.l_adv + self.l_x_rec + self.l_cls_xt + self.l_cls_xr + self.l_cls_rand + self.l_sem
        l_gen_total.backward()

        self.G_opt.step()

        ### update g_test ###
        this_model = self.G
        update_average(this_model.gen_test, this_model.gen)

def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)