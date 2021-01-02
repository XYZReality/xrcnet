'''
Train ncnet with strong loss
'''
import os
import numpy as np
import numpy.random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from lib.dataset_strong_loss import StrongLossDataset
from lib.loss import SparseStrongWeakLoss

from torch.utils.data import DataLoader
from lib.model_v2 import ImMatchNet
from torch.utils.tensorboard import SummaryWriter

from lib.tools import parseConfig
from lib.tools import ignorePath

import argparse

import sys
import shutil

# set visible gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

# Argument parsing
parser = argparse.ArgumentParser(description='X Resolution Correspondence Network Training script')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--training_file', type=str, default='../storage/MegaDepth_v1_SfM/training_pairs.txt')
parser.add_argument('--validation_file', type=str, default='../storage/MegaDepth_v1_SfM/validation_pairs.txt')
parser.add_argument('--image_path', type=str, default='../storage/MegaDepth_v1_SfM')
parser.add_argument('--result_model_fn', type=str, default='xrcnet', help='trained model filename')
parser.add_argument('--result_model_dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--config', type=str, default='configs/xrcnet.json', help='path to config file')
parser.add_argument('--no_code_backup', help='Dont backup code', action='store_true')
parser.add_argument('--no_timestamp_folder', help='Dont save to timestamp folder', action='store_true')
parser.add_argument('--log', help='Log to FILE in result_model_dir; use - for stdout (default is log.txt)', metavar='FILE', default='log.txt')
args = parser.parse_args()
print(args)

cfg = parseConfig(args.config)

# directory responsible for keeping a copy of the training, including:
# - config file (hyperparams)
# - code to reproduce training 
# - log file (iterations, loss report, etc)
if not args.no_timestamp_folder:
    time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    setting = "{}_backbone={}_gauKer={}_mode={}_loss={}_lr={}".format(os.path.basename(args.config)[:-5], cfg.backbone, cfg.gauss_size, cfg.loss["mode"], cfg.loss["loss"], cfg.optimizer["lr"])
    root_folder = os.path.join(args.result_model_dir, '%s_%s_%s_%d' % (args.result_model_fn, setting, time_string, os.getpid()))
else:
    root_folder = args.result_model_dir
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

# backup all the code for this training
if not args.no_code_backup:
    code_folder = os.path.abspath(os.path.dirname(__file__))
    shutil.copytree(code_folder, os.path.join(root_folder, os.path.basename(code_folder)), ignore=shutil.ignore_patterns("trained_models"))

# store the output to a log file
if args.log != '-':
    sys.stdout = open(os.path.join(root_folder, args.log), 'w')

im_fe_ratio = cfg.im_fe_ratio
model = ImMatchNet(use_cuda=use_cuda, multi_gpu=cfg.training["multi_gpu"], ncons_kernel_sizes=cfg.NCNet["kernel_sizes"],
                   ncons_channels=cfg.NCNet["channels"], checkpoint=args.checkpoint, feature_extraction_cnn=cfg.backbone,
                   nc_learnable=cfg.NCNet["learned"])
if cfg.training["multi_gpu"]:
    model = model.cuda()
    model = nn.DataParallel(model)


# Set which parts of the model to train
if cfg.training["fe_finetune_params"]>0:
    for i in range(cfg.training["fe_finetune_params"]):
        for p in model.FeatureExtraction.model[-1][-(i+1)].parameters():
            p.requires_grad=True

print('Trainable parameters:')
n = 0
for i,(name,p) in enumerate(filter(lambda p: p[1].requires_grad, model.named_parameters())):
    print("[{}] {}: {}".format(i+1, name, p.shape))
    n+=np.prod(p.shape)
print("Total params: {}".format(n))

if cfg.optimizer["type"].lower() == "adam":
    print('Using Adam optimizer')
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.optimizer["lr"])
elif cfg.optimizer["type"].lower() == "sgd":
    print('Using SGD optimizer')
    optimizer = optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.optimizer["lr"], momentum=cfg.optimizer["momentum"])

if cfg.optimizer["use_scheduler"]:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.optimizer["scheduler_milestone"], gamma=0.5)

# Define checkpoint_name
ckpt_dir = os.path.join(root_folder, "ckpts")
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
checkpoint_name = os.path.join(ckpt_dir, datetime.datetime.now().strftime(
                               "%Y-%m-%d_%H_%M") + '_' + args.result_model_fn + '_%s_gauKer=%d_mode=%d' %
                               (cfg.backbone,cfg.gauss_size, cfg.loss["mode"]) +'.pth.tar')

# build transform
transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_set = StrongLossDataset(file=args.training_file, image_path=args.image_path, transforms=transformer)
validation_set = StrongLossDataset(file=args.validation_file, image_path=args.image_path, transforms=transformer)

# build dataloader
training_loader = DataLoader(training_set, batch_size=cfg.training["batch_size"], num_workers=cfg.training["num_workers"], shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=cfg.training["batch_size"], num_workers=cfg.training["num_workers"], shuffle=True)
args
if cfg.training["multi_gpu"]:
    model.module.FeatureExtraction.eval()
else:
    model.FeatureExtraction.eval()

# create Tensorboard writer
if cfg.training["use_writer"]:
    writer = SummaryWriter(os.path.join(root_folder,'tensorboard/MegaDepth/' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")+'_'+args.result_model_fn))

# create strongly supervised loss
loss_fn = SparseStrongWeakLoss ( image_size = cfg.image_size, model = model, loss_name = cfg.loss["loss"], backbone=cfg.backbone,
                     weight_orthogonal=cfg.loss["weight_orthogonal"], weight_loss = cfg.loss["weight_loss"], fine_coarse_ratio=cfg.fine_coarse_ratio,
                     im_fe_ratio = im_fe_ratio, gauss_size = cfg.gauss_size, mode = cfg.loss["mode"], N=cfg.numKey)

best = float("inf")
PCK_best = 0
for epoch in range(cfg.training["num_epochs"]):
    epoch = epoch+1
    running_loss = 0

    for idx, batch in enumerate(training_loader):
        batch['source_image'] = batch['source_image'].cuda()
        batch['target_image'] = batch['target_image'].cuda()
        batch['source_points'] = batch['source_points'].cuda()
        batch['target_points'] = batch['target_points'].cuda()
        batch['assignment'] = batch['assignment'].cuda()
        optimizer.zero_grad()
        loss, _, _, _, _ = loss_fn(batch)

        loss.backward()
        optimizer.step()
        loss_item = loss.item()
        print('epoch', epoch, 'batch', idx, 'batch training loss', loss_item, 'lr', optimizer.param_groups[0]['lr'])
        running_loss += loss_item
        if cfg.training["use_writer"]:
            writer.add_scalar('training_loss', loss_item, (epoch-1) * len(training_loader) + idx)

        sys.stdout.flush()

    train_mean_loss = running_loss / len(training_loader)

    with torch.no_grad():
        running_PCK = 0
        running_loss = 0
        # model.eval()
        for idx, batch in enumerate(validation_loader):
            batch['source_image'] = batch['source_image'].cuda()
            batch['target_image'] = batch['target_image'].cuda()
            batch['source_points'] = batch['source_points'].cuda()
            batch['target_points'] = batch['target_points'].cuda()
            batch['assignment'] = batch['assignment'].cuda()

            loss, _, _, _, _ = loss_fn(batch)

            loss_item = loss.item()
            running_loss += loss_item

        val_mean_loss = running_loss / len(validation_loader)

    is_best = val_mean_loss < best

    if is_best:
        best = val_mean_loss

    print('validation_loss', val_mean_loss)
    if cfg.training["use_writer"]:
        writer.add_scalar('validation_loss', val_mean_loss, epoch-1)
    if cfg.optimizer["use_scheduler"]:
        scheduler.step()
    dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'args': args,
        'optimizer': optimizer.state_dict(),
        'training_loss': train_mean_loss,
        'validation_loss': val_mean_loss
    }

    dirname = os.path.dirname(checkpoint_name)
    basename = os.path.basename(checkpoint_name)

    print('is best?', is_best)
    if is_best:
        print('saving best model...')
        torch.save(dict, os.path.join(dirname, 'best_'+basename))

    sys.stdout.flush()
