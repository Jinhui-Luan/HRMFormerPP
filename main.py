import os
import time
import random
import IPython
import pymesh
import numpy as np
from tqdm import tqdm
from math import cos, pi
from chamferdist import ChamferDistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel 
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, AdamW

import transformer
from transformer import SMPLModel_torch
from option import BaseOptionParser
from plyfile import PlyData, PlyElement


class MyDataset(Dataset):
    def __init__(self, marker, theta, beta, joint, theta_max, theta_min):
        self.marker = marker
        self.theta = theta
        self.beta = beta
        self.joint = joint
        self.theta_max = theta_max
        self.theta_min = theta_min

    def __len__(self):
        return len(self.marker)

    def __getitem__(self, index):
        return {
            'marker': self.marker[index],
            'theta': self.theta[index],
            'beta': self.beta[index],
            'joint': self.joint[index],
            'theta_max': self.theta_max,
            'theta_min': self.theta_min
        }


class WarmUpScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def batch_step(self):
        "Step with the inner optimizer"
        self.update_lr()
        self.optimizer.step()

    def get_lr(self):
        n_steps, n_warmup_steps, d_model, lr_mul = self.n_steps, self.n_warmup_steps, self.d_model, self.lr_mul
        return lr_mul * (d_model ** (-0.5)) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def update_lr(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def epoch_step(self):
        return


class StepScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, base_lr, step_epoch, total_epoch, clip=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_epoch = step_epoch
        self.total_epoch = total_epoch
        self.clip = clip
        self.epoch = 1

    def batch_step(self):
        self.optimizer.step()

    def get_lr(self):
        if self.epoch < self.step_epoch:
            lr = self.base_lr
        elif self.epoch < self.step_epoch * 3:
            lr = self.base_lr / 10
        else:
            lr = self.base_lr / 100

        return lr

    def update_lr(self):
        ''' Learning rate scheduling per step '''
        self.epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def epoch_step(self):
        self.update_lr()


class CosineScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, base_lr, step_epoch, total_epoch, clip=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_epoch = step_epoch
        self.total_epoch = total_epoch
        self.clip = clip
        self.epoch = 1

    def batch_step(self):
        self.optimizer.step()

    def get_lr(self):
        if self.epoch < self.step_epoch:
            lr = self.base_lr
        else:
            lr = self.clip + 0.5 * (self.base_lr - self.clip) * \
                (1 + cos(pi * ((self.epoch - self.step_epoch) / (self.total_epoch - self.step_epoch))))
        return lr

    def update_lr(self):
        ''' Learning rate scheduling per step '''
        self.epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
    def epoch_step(self):
        self.update_lr()


def init_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        # nn.init.constant_(m.bias, 0)


def attach_placeholder(X):
    ''' add <start> and <end> placeholder '''
    placeholder = torch.zeros((X.shape[0], 1, X.shape[2]), dtype=torch.float32, device=X.device, requires_grad=True)
    return torch.cat((placeholder, X), dim=1)


def get_data_loader(data_path, exp_path, batch_size, mode, m, f, stride, rs):
    data = torch.load(os.path.join(data_path, mode + '_' + str(m) + '.pt'))
    print('Successfully load data from ' + mode + '_' + str(m) + '.pt!')
    
    marker = torch.Tensor(data['marker']).reshape(-1, f, m, 3).to(torch.float32)              # (n_seq, f, m, 3)
    theta = torch.Tensor(data['theta']).reshape(-1, f, 24, 3).to(torch.float32)               # (n_seq, f, j, 3)
    beta = torch.Tensor(data['beta']).reshape(-1, f, 10).to(torch.float32)                    # (n_seq, f, 10)
    # vertex = torch.Tensor(data['vertex']).reshape(-1, f, 6890, 3).to(torch.float32)           # (n_seq, f, v, 3)
    joint = torch.Tensor(data['joint']).reshape(-1, f, 24, 3).to(torch.float32)               # (n_seq, f, j, 3)
    theta_max = torch.Tensor(data['theta_max']).to(torch.float32)                                       # (j, 3)
    theta_min = torch.Tensor(data['theta_min']).to(torch.float32)                                       # (j, 3)

    # for i in range(marker.shape[0]):
    #     marker[i, :, :, :] = marker[i, :, torch.randperm(marker.shape[2]), :]
    if rs:
        l = marker.shape[0]
        idx = torch.randint(l, [l, ])[:l//stride]
        marker = marker[idx]
        theta = theta[idx]
        beta = beta[idx]
        # vertex = vertex[idx]
        joint = joint[idx]
    else:
        marker = marker[::stride]
        theta = theta[::stride]
        beta = beta[::stride]
        joint = joint[::stride]

    print('{} dataset shape: {}.'.format(mode, marker.shape).capitalize())
    with open(os.path.join(exp_path, 'parameters.txt'), 'a') as f:
        f.write('{} dataset shape: {}.\n'.format(mode, marker.shape).capitalize())

    dataset = MyDataset(marker, theta, beta, joint, theta_max, theta_min)
    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size, shuffle=False)

    return dataloader


def print_performances(header, loss, mpjpe, mpvpe, jitter, lr, start_time):
    print(' - {:12} loss: {:6.4f}, mpjpe: {:6.4f}, mpvpe: {:6.4f}, jitter: {:6.4f}, lr: {:10.8f}, time cost: {:4.2f} min'.format(
        f"({header})", loss, mpjpe, mpvpe, jitter, lr, (time.time()-start_time)/60))


def load_checkpoint(model, args, device, start_epoch, scheduler=None):
    if hasattr(model, 'module'):
        model = model.module
    model_path = os.path.join(args.model_save_path, 'model_' + str(start_epoch) + '.chkpt')
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(checkpoint['model'])

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 1

    # load optimizer
    if scheduler is not None:
        assert 'optimizer' in checkpoint
        scheduler.optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.epoch = epoch
        
    return epoch


def weighted_data_loss(y_pred, y_gt, rate, criterion):
    '''
    y_pred & y_gt: pose parameters theta of SMPL model with size of (bs, 24, 3)
    rate: the rate of the weight between parent and child nodes
    criterion: the criterion for calculating loss, e.g. MESLoss
    return: the data term of loss
    '''
    _, j, _ = y_pred.shape
    root = [0, 3, 6, 9]
    child1 = [1, 2, 12, 13, 14, 16, 17]
    child2 = [4, 5, 15, 18, 19]
    child3 = [7, 8, 10, 11, 20, 21, 22, 23]

    loss = 0
    for i, part in enumerate([root, child1, child2, child3]):
        # print(i, part, rate ** i)
        for idx in part:
            # print(idx)
            l = criterion(y_pred[:, idx, :], y_gt[:, idx, :]) * rate ** i
            # IPython.embed()
            loss += l
        
    return loss / j


def regularization_loss(y, max, min):
    '''
    y: the predicted pose parameters theta of SMPL model with size of (bs, f, 24, 3)
    max (min): the maximum (minimum) of theta in dataset with size of (bs, 24, 3)
    return: the regularization term of loss
    '''
    bs, f, j, _ = y.shape
    zero = torch.zeros_like(y)
    max_unsqueeze = max.unsqueeze(1).repeat(1, f, 1, 1)
    min_unsqueeze = min.unsqueeze(1).repeat(1, f, 1, 1)

    return (torch.maximum(zero, y-max_unsqueeze).sum() + torch.maximum(zero, min_unsqueeze-y).sum()) / (bs * f * j)


def chamfer_distance_loss(pc1, pc2, criterion_cd, bidirectional=True):
    '''
    pc1 & pc2: the point cloud used to compute chamfer distance with size of (bs, n, 3)
    criterion_cd: the ChamferDistance object
    return: the chamfer distance  loss
    '''
    _, n1, _ = pc1.shape
    _, n2, _ = pc2.shape
    loss = criterion_cd(pc1, pc2, bidirectional=bidirectional)

    return loss / (n1 + n2)


def temporal_smooth_loss1(x, rate, criterion):
    '''
    x: the predicted pose parameter or joint postion with size of (bs, f, 24, 3)
    rate: the rate of the weight between parent and child nodes
    criterion: the criterion for calculating loss, e.g. nn.MESLoss
    return: the temporal smooth term of loss
    '''
    _, f, j, _ = x.shape
    root = [0, 3, 6, 9]
    child1 = [1, 2, 12, 13, 14, 16, 17]
    child2 = [4, 5, 15, 18, 19]
    child3 = [7, 8, 10, 11, 20, 21, 22, 23]

    loss = 0
    for i in range(f-1):
        for j, part in enumerate([root, child1, child2, child3]):
            # print(i, part, rate ** i)
            for idx in part:
                # print(idx)
                l = criterion(x[:, i+1, idx, :], x[:, i, idx, :]) * rate ** j
                # IPython.embed()
                loss += l
        
    return loss / ((f-1) * j)


def temporal_smooth_loss2(x, criterion):
    '''
    x: the position of predicted joint or vertex with size of (bs, f, n, 3)
    rate: the rate of the weight between parent and child nodes
    criterion: the criterion for calculating loss, e.g. nn.MESLoss
    return: the temporal smooth term of loss
    '''
    _, f, n, _ = x.shape

    loss = 0
    for i in range(1, f-1):
        x_ave = (x[:, i-1, :, :] + x[:, i+1, :, :]) / 2
        l = criterion(x[:, i, :, :], x_ave)
        loss += l
        
    return loss / ((f-2) * n)


def get_speed(x):
    '''
    x: the position of input marker or predicted joint or vertex with size of (bs, f, n, 3)
    return: the average speed of x with size of (bs, f-1, n, 3)
    '''

    return x[:, 1:, :, :] - x[:, 0:-1, :, :]


def temporal_smooth_loss3(pc1, pc2, criterion):
    '''
    pc1: the position of input marker with size of (bs, f, m, 3)
    pc2: the position of predicted joint or vertex with size of (bs, f, n, 3)
    criterion: the criterion for calculating loss, e.g. nn.MESLoss
    return: the temporal smooth term of loss
    '''
    speed1 = get_speed(pc1)             # (bs, f-1, m, 3)
    speed2 = get_speed(pc2)             # (bs, f-1, n, 3)
    acc1 = get_speed(speed1)            # (bs, f-2, m, 3)
    acc2 = get_speed(speed2)            # (bs, f-2, n, 3)

    return criterion(acc1.pow(2).sum(dim=-1).sqrt().mean(dim=-1), acc2.pow(2).sum(dim=-1).sqrt().mean(dim=-1)) 


def get_jitter(x):
    '''
    x: the position of predicted vertex with size of (bs, f, n, 3)
    return: the average jitter of x with size of (bs, f-3, n, 3)
    ''' 
    speed = get_speed(x)                # (bs, f-1, n, 3)
    acc = get_speed(speed)              # (bs, f-2, n, 3)
    jitter = get_speed(acc)             # (bs, f-3, n, 3)

    return jitter


def train(model, dataloader_train, dataloader_val, scheduler, device, args):
    criterion_mse = nn.MSELoss().to(device)
    criterion_cd = ChamferDistance().to(device)
    smpl_model_path = os.path.join(args.data_path, 'model_m.pkl')   
    smpl_model = SMPLModel_torch(smpl_model_path, device) 

    # train from scratch or continue
    if args.resume:
        start_epoch = load_checkpoint(model, args, device, args.start_epoch, scheduler=scheduler)
        print(' - [Info] Successfully load pretrained model, start epoch =', start_epoch)
    else:
        start_epoch = 1
        model.apply(weight_init)  

    # use tensorboard to plot curves
    if not args.no_tb:
        writer = SummaryWriter(os.path.join(args.log_save_path, 'tb'))

    log_train_file = os.path.join(args.log_save_path, 'train.log')
    log_valid_file = os.path.join(args.log_save_path, 'valid.log')

    print(' - [Info] Training performance will be written to file: {} and {}'.format(
          log_train_file, log_valid_file))

    if not args.resume:
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('{:6}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}\n'.format(
                'epoch', 'l', 'l_d', 'l_j', 'l_v', 'l_reg', 'l_cd', 'l_ts1', 'l_ts2', 'mpjpe', 'mpvpe', 'jitter', 'lr'))
            log_vf.write('{:6}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}, {:10}\n'.format(
                'epoch', 'l', 'l_d', 'l_j', 'l_v', 'l_reg', 'l_cd', 'l_ts1', 'l_ts2', 'mpjpe', 'mpvpe', 'jitter', 'lr'))

    val_metrics = []
    
    for i in range(start_epoch, args.total_epoch+1):
        print('[ Epoch', i, ']')

        # train epoch
        start = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_l, train_ld, train_lj, train_lv, train_lreg, train_lcd, train_lts1, train_lts2 ,train_mpjpe, train_mpvpe, train_jitter = train_epoch(
            model, smpl_model, dataloader_train, scheduler, criterion_mse, criterion_cd, device, args)
        print_performances('Training', train_l, train_mpjpe, train_mpvpe, train_jitter, lr, start)

        # validation epoch
        start = time.time()
        val_l, val_ld, val_lj, val_lv, val_lreg, val_lcd, val_lts1, val_lts2, val_mpjpe, val_mpvpe, val_jitter = val_epoch(
            model, smpl_model, dataloader_val, criterion_mse, criterion_cd, device, args)
        print_performances('Validation', val_l, val_mpjpe, val_mpvpe, val_jitter, lr, start)

        val_metric = 0.5 * val_mpjpe + 0.5 * val_mpvpe
        val_metrics.append(val_metric)

        checkpoint = {'epoch': i, 'settings': args, 'model': model.state_dict(), 'optimizer': scheduler.optimizer.state_dict()}

        if val_metric <= min(val_metrics):
            torch.save(checkpoint, os.path.join(args.model_save_path, 'model_best.chkpt'))
            print(' - [Info] The best model file has been updated.')
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(' - [Info] The best model file has been updated.\n')
                log_vf.write(' - [Info] The best model file has been updated.\n')
                
        if i % args.interval == 0:
            torch.save(checkpoint, os.path.join(args.model_save_path, 'model_' + str(i) +'.chkpt'))
            print(' - [Info] The model file has been saved for every {} epochs.'.format(args.interval))

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{:6d}: {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.8f}\n'.format(
                i, train_l, train_ld, train_lj, train_lv, train_lreg, train_lcd, train_lts1, train_lts2, train_mpjpe, train_mpvpe, train_jitter, lr))
            log_vf.write('{:6d}: {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.8f}\n'.format(
                i, val_l, val_ld, val_lj, val_lv, val_lreg, val_lcd, val_lts1, val_lts2, val_mpjpe, val_mpvpe, val_jitter, lr))

        if not args.no_tb:
            writer.add_scalars('l',{'train': train_l, 'val': val_l}, i)
            writer.add_scalars('l_d',{'train': train_ld, 'val': val_ld}, i)
            writer.add_scalars('l_j',{'train': train_lj, 'val': val_lj}, i)
            writer.add_scalars('l_v',{'train': train_lv, 'val': val_lv}, i)
            writer.add_scalars('l_reg',{'train': train_lreg, 'val': val_lreg}, i)
            writer.add_scalars('l_cd',{'train': train_lcd, 'val': val_lcd}, i)
            writer.add_scalars('l_ts1',{'train': train_lts1, 'val': val_lts1}, i)
            writer.add_scalars('l_ts2',{'train': train_lts2, 'val': val_lts2}, i)
            writer.add_scalars('mpjpe',{'train': train_mpjpe, 'val': val_mpjpe}, i)
            writer.add_scalars('mpvpe',{'train': train_mpvpe, 'val': val_mpvpe}, i)
            writer.add_scalars('jitter',{'train': train_jitter, 'val': val_jitter}, i)
            writer.add_scalar('lr', lr, i)


def train_epoch(model, smpl_model, dataloader_train, scheduler, criterion_mse, criterion_cd, device, args):
    model.train()
    loss = []
    loss_d = []                     # data term of loss function
    loss_j = []                     # joint term of loss function
    loss_v = []                     # vertex term of loss function
    loss_reg = []                   # regularization term of loss function
    loss_cd = []                    # chamferdistance between input marker and output mesh vertex
    loss_ts1 = []                   # temporal smooth term of loss function
    loss_ts2 = []                   # temporal smooth term of loss function
    MPJPE = []
    MPVPE = []
    JITTER = []

    desc = ' - (Training)   '
    for data in tqdm(dataloader_train, mininterval=2, desc=desc, leave=False, ncols=100):
        marker = data['marker'].to(device)                          # (bs, f, m, 3)
        theta = data['theta'].to(device)                            # (bs, f, spa_n_q, 3)
        beta = data['beta'].to(device)                              # (bs, f, 10)
        theta_max = data['theta_max'].to(device)                    # (bs, spa_n_q, 3)
        theta_min = data['theta_min'].to(device)                    # (bs, spa_n_q, 3)
        # vertex = data['vertex'].to(device)                          # (bs, f, 6890, 3)
        # joint = data['joint'].to(device)                            # (bs, f, 24, 3)

        bs, f, m, _ = marker.shape
        scheduler.optimizer.zero_grad()
        theta_pred = model(marker)                                  # (bs, f, spa_n_q, 3)

        smpl_model(beta.reshape(bs*f, 10), theta_pred.reshape(bs*f, args.spa_n_q, 3))
        joint_pred = smpl_model.joints.to(torch.float32)            # (bs*f, 24, 3)   
        vertex_pred = smpl_model.verts.to(torch.float32)            # (bs*f, 6890, 3)

        smpl_model(beta.reshape(bs*f, 10), theta.reshape(bs*f, args.spa_n_q, 3))
        joint= smpl_model.joints.to(torch.float32)                  # (bs*f, 24, 3)
        vertex = smpl_model.verts.to(torch.float32)                 # (bs*f, 6890, 3)

        mpjpe = (joint_pred - joint).pow(2).sum(dim=-1).sqrt().mean()
        mpvpe = (vertex_pred - vertex).pow(2).sum(dim=-1).sqrt().mean()
        jitter = get_jitter(vertex_pred.reshape(bs, f, 6890, 3)).pow(2).sum(dim=-1).sqrt().mean()

        # l_d = args.lambda1 * criterion(theta_pred, theta)
        l_d = args.lambda1 * weighted_data_loss(theta_pred.reshape(bs*f, args.spa_n_q, 3), theta.reshape(bs*f, args.spa_n_q, 3), args.rate, criterion_mse)
        l_j = args.lambda2 * abs((joint_pred - joint)).sum(dim=-1).mean()
        l_v = args.lambda3 * abs((vertex_pred - vertex)).sum(dim=-1).mean()
        l_reg = args.lambda4 * regularization_loss(theta_pred, theta_max, theta_min)
        l_cd = args.lambda5 * chamfer_distance_loss(marker.reshape(bs*f, m, 3), vertex_pred, criterion_cd)

        # temproal smooth loss function 1
        if args.lts == 1:
            l_ts1 = args.lambda6 * temporal_smooth_loss1(joint_pred.reshape(bs, f, 24, 3), args.rate, criterion_mse)
            l_ts2 = args.lambda7 * temporal_smooth_loss1(theta_pred, args.rate, criterion_mse)
        # temporal smooth loss function 2
        if args.lts == 2:
            l_ts1 = args.lambda6 * temporal_smooth_loss2(joint_pred.reshape(bs, f, 24, 3), criterion_mse)
            l_ts2 = args.lambda7 * temporal_smooth_loss2(vertex_pred.reshape(bs, f, 6890, 3), criterion_mse)
        # temporal smooth loss function 3
        if args.lts == 3:
            l_ts1 = args.lambda6 * temporal_smooth_loss3(marker, joint_pred.reshape(bs, f, 24, 3), criterion_mse)
            l_ts2 = args.lambda7 * temporal_smooth_loss3(marker, vertex_pred.reshape(bs, f, 6890, 3), criterion_mse)

        l = l_d + l_j + l_v + l_reg + l_cd + l_ts1 + l_ts2

        # IPython.embed()

        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scheduler.batch_step()

        loss.append(l)
        loss_d.append(l_d)
        loss_j.append(l_j)
        loss_v.append(l_v)
        loss_reg.append(l_reg)
        loss_cd.append(l_cd)
        loss_ts1.append(l_ts1)
        loss_ts2.append(l_ts2)

        MPJPE.append(mpjpe.clone().detach())
        MPVPE.append(mpvpe.clone().detach())
        JITTER.append(jitter.clone().detach())
    
    scheduler.epoch_step()
        
    return torch.Tensor(loss).mean(), torch.Tensor(loss_d).mean(), torch.Tensor(loss_j).mean(), torch.Tensor(loss_v).mean(), \
        torch.Tensor(loss_reg).mean(), torch.Tensor(loss_cd).mean(), torch.Tensor(loss_ts1).mean(), torch.Tensor(loss_ts2).mean(), \
        torch.Tensor(MPJPE).mean(), torch.Tensor(MPVPE).mean(), torch.Tensor(JITTER).mean()


def val_epoch(model, smpl_model, dataloader_val, criterion_mse, criterion_cd, device, args):
    model.eval()
    loss = []
    loss_d = []
    loss_j = []
    loss_v = []
    loss_reg = []
    loss_cd = []
    loss_ts1 = []
    loss_ts2 = []
    MPJPE = []
    MPVPE = []
    JITTER = []

    desc = ' - (Validation) '
    with torch.no_grad():
        for data in tqdm(dataloader_val, mininterval=2, desc=desc, leave=False, ncols=100):
            marker = data['marker'].to(device)
            theta = data['theta'].to(device)
            beta = data['beta'].to(device)
            theta_max = data['theta_max'].to(device)                 
            theta_min = data['theta_min'].to(device)                  
            # vertex = data['vertex'].to(device)
            # joint = data['joint'].to(device)

            bs, f, m, _ = marker.shape

            theta_pred = model(marker)

            smpl_model(beta.reshape(bs*f, 10), theta_pred.reshape(bs*f, args.spa_n_q, 3))
            joint_pred = smpl_model.joints.to(torch.float32) 
            vertex_pred = smpl_model.verts.to(torch.float32) 

            smpl_model(beta.reshape(bs*f, 10), theta.reshape(bs*f, args.spa_n_q, 3))
            joint= smpl_model.joints.to(torch.float32)       
            vertex = smpl_model.verts.to(torch.float32) 

            mpjpe = (joint_pred - joint).pow(2).sum(dim=-1).sqrt().mean()
            mpvpe = (vertex_pred - vertex).pow(2).sum(dim=-1).sqrt().mean()
            jitter = get_jitter(vertex_pred.reshape(bs, f, 6890, 3)).pow(2).sum(dim=-1).sqrt().mean()

            # l_d = args.lambda1 * criterion(theta_pred, theta)
            l_d = args.lambda1 * weighted_data_loss(theta_pred.reshape(bs*f, args.spa_n_q, 3), theta.reshape(bs*f, args.spa_n_q, 3), args.rate, criterion_mse)
            l_j = args.lambda2 * abs((joint_pred - joint)).sum(dim=-1).mean()
            l_v = args.lambda3 * abs((vertex_pred - vertex)).sum(dim=-1).mean()
            l_reg = args.lambda4 * regularization_loss(theta_pred, theta_max, theta_min)
            l_cd = args.lambda5  * chamfer_distance_loss(marker.reshape(bs*f, m, 3), vertex_pred, criterion_cd)

            # temproal smooth loss function 1
            if args.lts == 1:
                l_ts1 = args.lambda6 * temporal_smooth_loss1(joint_pred.reshape(bs, f, 24, 3), args.rate, criterion_mse)
                l_ts2 = args.lambda7 * temporal_smooth_loss1(theta_pred, args.rate, criterion_mse)
            # temporal smooth loss function 2
            if args.lts == 2:
                l_ts1 = args.lambda6 * temporal_smooth_loss2(joint_pred.reshape(bs, f, 24, 3), criterion_mse)
                l_ts2 = args.lambda7 * temporal_smooth_loss2(vertex_pred.reshape(bs, f, 6890, 3), criterion_mse)
            # temporal smooth loss function 3
            if args.lts == 3:
                l_ts1 = args.lambda6 * temporal_smooth_loss3(marker, joint_pred.reshape(bs, f, 24, 3), criterion_mse)
                l_ts2 = args.lambda7 * temporal_smooth_loss3(marker, vertex_pred.reshape(bs, f, 6890, 3), criterion_mse)

            l = l_d + l_j + l_v + l_reg + l_cd + l_ts1 + l_ts2
            
            loss.append(l)
            loss_d.append(l_d)
            loss_j.append(l_j)
            loss_v.append(l_v)
            loss_reg.append(l_reg)
            loss_cd.append(l_cd)
            loss_ts1.append(l_ts1)
            loss_ts2.append(l_ts2)

            MPJPE.append(mpjpe.clone().detach())
            MPVPE.append(mpvpe.clone().detach())
            JITTER.append(jitter.clone().detach())

    return torch.Tensor(loss).mean(), torch.Tensor(loss_d).mean(), torch.Tensor(loss_j).mean(), torch.Tensor(loss_v).mean(), \
        torch.Tensor(loss_reg).mean(), torch.Tensor(loss_cd).mean(), torch.Tensor(loss_ts1).mean(), torch.Tensor(loss_ts2).mean(), \
        torch.Tensor(MPJPE).mean(), torch.Tensor(MPVPE).mean(), torch.Tensor(JITTER).mean()
    

def write_ply(save_path, vertex, rgb=None):
    """
    Paramerer:
    ---------
    save_path : path to save
    vertex: point cloud with size of (n_points, 3)
    rgb: RGB information of each point with size of (n_point, 3)
    """

    vertex = [(vertex[i, 0], vertex[i, 1], vertex[i, 2]) for i in range(vertex.shape[0])]
    vertex = PlyElement.describe(np.array(vertex, dtype=[('x', 'float32'), ('y', 'float32'), ('z', 'float32')]), 'vertex')
    
    PlyData([vertex]).write(save_path)   


def write_mesh(save_path, vertex, face, rgb=None):
    """
    Paramerer:
    ---------
    save_path : path to save
    vertex: point cloud with size of (n_points, 3)
    face: vertex connectivity of SMPL mesh
    rgb: RGB information of each point with size of (n_point, 3)
    """
    mesh_ref = pymesh.load_mesh("./template_color.ply")

    face.dtype='int32'
    mesh = pymesh.form_mesh(vertices=vertex, faces=face)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(save_path, mesh, "red", "green", "blue", ascii=True)


def test(model, dataloader_test, device, args):
    # criterion = nn.MSELoss().to(device)
    smpl_model_path = os.path.join(args.data_path, 'model_m.pkl')   
    smpl_model = SMPLModel_torch(smpl_model_path, device) 
    
    # print(face, face.shape)
    # IPython.embed()

    model.eval()
    # loss = []
    # loss_data = []
    # loss_joint = []
    # loss_vertex = []
    MPJPE = []
    MPVPE = []
    JITTER = []
    batch = 0

    desc = ' -       (Test) '
    with torch.no_grad():
        for data in tqdm(dataloader_test, mininterval=2, desc=desc, leave=False, ncols=100):
            marker = data['marker'].to(device)
            theta = data['theta'].to(device)
            beta = data['beta'].to(device)
            # vertex = data['vertex'].to(device)
            # joint = data['joint'].to(device)

            bs, f, m, _ = marker.shape
            theta_pred = model(marker)                                  # (bs, f, spa_n_q, 3)

            smpl_model(beta.reshape(bs*f, 10), theta_pred.reshape(bs*f, args.spa_n_q, 3))
            joint_pred = smpl_model.joints.to(torch.float32)            # (bs*f, 24, 3)   
            vertex_pred = smpl_model.verts.to(torch.float32)            # (bs*f, 6890, 3)

            smpl_model(beta.reshape(bs*f, 10), theta.reshape(bs*f, args.spa_n_q, 3))
            joint= smpl_model.joints.to(torch.float32)                  # (bs*f, 24, 3)
            vertex = smpl_model.verts.to(torch.float32)                 # (bs*f, 6890, 3)

            mpjpe = (joint_pred - joint).pow(2).sum(dim=-1).sqrt().mean()
            mpvpe = (vertex_pred - vertex).pow(2).sum(dim=-1).sqrt().mean()
            jitter = get_jitter(vertex_pred.reshape(bs, f, 6890, 3)).pow(2).sum(dim=-1).sqrt().mean()

            MPJPE.append(mpjpe.clone().detach())
            MPVPE.append(mpvpe.clone().detach())
            JITTER.append(jitter.clone().detach())

            if args.visualize:
                face = smpl_model.faces

                for i in range(marker.shape[0]):
                    # generate rgb color
                    rgb_marker = np.repeat(np.array([[255, 0, 0]]), marker.shape[1], axis=0)        # show marker in red 
                    rgb_joint = np.repeat(np.array([[0, 255, 0]]), joint.shape[1], axis=0)          # show joint in green
                    rgb_vertex = np.repeat(np.array([[123, 123, 123]]), vertex.shape[1], axis=0)    # show vertex in gray             

                    # concatenate the points of mesh and markers
                    m = marker[i].to('cpu')
                    j = joint[i].to('cpu')
                    j_pred = joint_pred[i].to('cpu')
                    v = vertex[i].to('cpu')
                    v_pred = vertex_pred[i].to('cpu')
                    
                    # point = np.vstack((j, v))
                    # point_pred = np.vstack((j_pred, v_pred))
                    # rgb = np.vstack((rgb_joint, rgb_vertex))

                    os.makedirs(os.path.join(args.vis_path, args.exp_name), exist_ok=True)
                    write_ply(os.path.join(args.vis_path, args.exp_name, str(batch) + '_' + str(i) + '_marker.ply'), m)
                    write_mesh(os.path.join(args.vis_path, args.exp_name, str(batch) + '_' + str(i) + '_mesh_gt.ply'), v, face)
                    write_mesh(os.path.join(args.vis_path, args.exp_name, str(batch) + '_' + str(i) + '_mesh.ply'), v_pred, face)
                    write_ply(os.path.join(args.vis_path, args.exp_name, str(batch) + '_' + str(i) + '_joint_gt.ply'), j)
                    write_ply(os.path.join(args.vis_path, args.exp_name, str(batch) + '_' + str(i) + '_joint.ply'), j_pred)

                batch += 1


    return torch.Tensor(MPJPE).mean(), torch.Tensor(MPVPE).mean(), torch.Tensor(JITTER).mean()


def main():
    # get parser of option.py
    parser = BaseOptionParser()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.d_h3 = args.d_model // 2
    args.d_h2 = args.d_h3 // 2
    args.d_h1 = args.d_h2 // 4
    args.d_ffn = args.d_model * 2

    if args.net == 1:
        model = transformer.Transformer1(args).to(device)
    elif args.net == 2:
        model = transformer.Transformer2(args).to(device)

    args.exp_path = os.path.join(args.output_path, args.exp_name)
    args.log_save_path = os.path.join(args.exp_path, 'logs')
    os.makedirs(args.log_save_path, exist_ok=True)
    args.model_save_path = os.path.join(args.exp_path, 'models')
    os.makedirs(args.model_save_path, exist_ok=True)

    # train mode
    if args.mode == 'train':
        # initialization
        if args.seed is not None:
            init_random_seed(args.seed)
            
        if not args.output_path:
            print('No experiment result will be saved.')
            raise

        print(args)
        parser.save(os.path.join(args.exp_path, 'parameters.txt'))

        dl_train = get_data_loader(args.data_path, args.exp_path, args.bs, 'train', args.m, args.f, args.stride, args.rs)
        dl_val = get_data_loader(args.data_path, args.exp_path, args.bs, 'val', args.m, args.f, 1, args.rs)

        with open(os.path.join(args.exp_path, 'parameters.txt'), 'a') as f:
            f.writelines('---------- model ---------' + '\n')
            f.write(str(model))
            f.writelines('----------- end ----------' + '\n')
        


        # create optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.base_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
        if args.optim == 'step':
            scheduler = StepScheduledOptim(optimizer, args.base_lr, args.step_epoch, args.total_epoch)
        elif args.optim == 'cosine':
            scheduler = CosineScheduledOptim(optimizer, args.base_lr, args.step_epoch, args.total_epoch)
        elif args.optim == 'warmup':
            scheduler = WarmUpScheduledOptim(optimizer, args.lr_mul, args.d_model, args.n_warmup_steps)
        
        # training
        train(model, dl_train, dl_val, scheduler, device, args)

    elif args.mode == 'test':
        model_path = os.path.join(args.output_path, args.exp_name, 'models', 'model_best.chkpt')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        print('Successfully load checkpoint of model!')
        dl_test = get_data_loader(args.data_path, args.exp_path, args.bs, 'test', args.m, args.f, 10, args.rs)
        mpjpe, mpvpe, jitter = test(model, dl_test, device, args)
        print(' - mpjpe: {:6.4f}, mpvpe: {:6.4f}, jitter: {:6.4f}'.format(mpjpe, mpvpe, jitter))
        with open(os.path.join(args.log_save_path, 'test.log'), 'a') as f:
            f.write(' - mpjpe: {:6.4f}, mpvpe: {:6.4f}, jitter: {:6.4f}'.format(mpjpe, mpvpe, jitter))


if __name__ == '__main__':
    ''' 
    Usage:
    CUDA_VISIBLE_DEVICES='0' python main.py -exp_name='1_net1_rp_211' -pre_norm -spa_emb -tem_emb
    '''

    main()

