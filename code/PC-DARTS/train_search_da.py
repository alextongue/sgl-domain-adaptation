'''
PC-DARTS train_search for domain adaptation
'''

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import time

from torch.autograd import Variable
from model_search import DANN
from architect import ArchitectDA
from dataset import get_train_dataset


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../../data', help='location of the data corpus')
parser.add_argument('--src_set', type=str, default='mnist', help='name of source dataset')
parser.add_argument('--tgt_set', type=str, default='mnistm', help='name of target dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# only support mnist, mnistm right now
assert args.src_set in [ 'mnist', 'mnistm' ]
assert args.tgt_set in [ 'mnist', 'mnistm' ]

NUM_CLASSES = 10
NUM_DOMAINS = 2

# threshold number of epochs after which architecture search
# begins
ARCH_EPOCH_THRESH = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    if device != 'cuda':
        logging.info( 'no gpu device available' )
        sys.exit( 1 )
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    
    # setup criterion, model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    label_criterion = nn.NLLLoss().to( device )
    domain_criterion = nn.NLLLoss().to( device )
    #model = NetworkFE(args.init_channels, NUM_CLASSES, args.layers, criterion)
    model = DANN(args.init_channels, NUM_CLASSES, args.layers,
            label_criterion, domain_criterion)
    model = model.to( device )
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # TODO: setup right optimizer
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    #optimizer = torch.optim.Adam(
    #        model.parameters(),
    #        args.learning_rate )
    
    src_train_data = get_train_dataset( args.src_set, args )
    tgt_train_data = get_train_dataset( args.tgt_set, args )

    num_train = min( len(src_train_data), len(tgt_train_data) ) // 1
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # DataLoader for src,tgt training data
    src_train_queue = torch.utils.data.DataLoader(
      src_train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=8) 
    tgt_train_queue = torch.utils.data.DataLoader(
      tgt_train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=8)
  
    # DataLoader for src,tgt validation data
    src_valid_queue = torch.utils.data.DataLoader(
      src_train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=8)
    tgt_valid_queue = torch.utils.data.DataLoader(
      tgt_train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=8)
  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  
    # Use architect for domain adaptation
    architect = ArchitectDA(model, args)

    # import pdb; pdb.set_trace()
    # main loop
    for epoch in range( args.epochs ):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # train step
        train_acc, train_obj = train( src_train_queue, tgt_train_queue,
                src_valid_queue, tgt_valid_queue,
                model, architect, label_criterion, 
                domain_criterion, optimizer, lr,epoch)
        logging.info('train_acc %f', train_acc)
    
        # validation only on last epoch
        if args.epochs-epoch<=1:
          valid_acc, valid_obj = infer( src_valid_queue,
                  tgt_valid_queue, model, label_criterion,
                  domain_criterion )
          logging.info('valid_acc %f', valid_acc)

        # save model
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        scheduler.step()
    
    print( 'Experiment Dir:', args.save )

def train( src_train_queue, tgt_train_queue,
        src_valid_queue, tgt_valid_queue,
        model, architect, label_criterion,
        domain_criterion, optimizer, 
        lr, epoch ):
    '''
    PC-DARTS training routine for domain adaptation
    '''
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    label_loss_meter = utils.AvgrageMeter()
    domain_loss_meter = utils.AvgrageMeter() 

    N = min( len( src_train_queue ), len( tgt_train_queue ) )
    src_train_queue_iter = iter( src_train_queue )
    tgt_train_queue_iter = iter( tgt_train_queue )
    src_valid_queue_iter = iter( src_valid_queue )
    tgt_valid_queue_iter = iter( tgt_valid_queue )
    # import pdb; pdb.set_trace()

    for step in range( N ):
        # zero out gradients
        optimizer.zero_grad()
        # compute alpha used in gradient reversal layer
        p = float(step + epoch * N) / args.epochs / N
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # get training data
        src_images, src_labels = next( src_train_queue_iter )
        tgt_images, _ = next( tgt_train_queue_iter )
        src_images, src_labels, tgt_images = src_images.to( device ), \
                src_labels.to( device ), tgt_images.to( device )
        batch_size = len( src_labels )

        if epoch >= ARCH_EPOCH_THRESH:
            # update architecture using validation data
            val_src_imgs, val_src_labels = next( src_valid_queue_iter )
            val_tgt_imgs, _ = next( tgt_valid_queue_iter )
            val_src_imgs, val_src_labels, val_tgt_imgs = \
                    val_src_imgs.to( device ), val_src_labels.to( device ), \
                    val_tgt_imgs.to( device )
            architect.step( val_src_imgs, val_src_labels, val_tgt_imgs,
                alpha )

        # train model using src_data
        src_domain = torch.zeros( batch_size ).long().to( device )
        src_labels_out, src_domain_out = model( src_images, alpha )
        src_label_loss = label_criterion( src_labels_out, src_labels )
        src_domain_loss = domain_criterion( src_domain_out, src_domain )

        # train model using tgt_data
        tgt_domain = torch.ones( batch_size ).long().to( device )
        _, tgt_domain_out = model( tgt_images, alpha )
        tgt_domain_loss = domain_criterion( tgt_domain_out, tgt_domain )

        # optimize
        domain_loss = src_domain_loss + tgt_domain_loss
        loss = src_label_loss + domain_loss
        loss.backward()
        optimizer.step()

        # calculate accuracy on src labels
        acc1, acc5 = utils.accuracy( src_labels_out, src_labels, topk=(1, 5) )

        # update stats
        objs.update( loss.item(), batch_size )
        top1.update( acc1.item(), batch_size )
        top5.update( acc5.item(), batch_size )
        label_loss_meter.update( src_label_loss.item(), batch_size )
        domain_loss_meter.update( domain_loss.item(), batch_size )

        # report stats
        if step % args.report_freq == 0:
            logging.info( 'train %03d %e %e %e %f %f', step, objs.avg,
                    label_loss_meter.avg, domain_loss_meter.avg,
                    top1.avg, top5.avg )
    
    return top1.avg, objs.avg

def infer( src_val_queue, tgt_val_queue, model,
        label_criterion, domain_criterion ):
    '''
    PC-DARTS inference routine for domain adaptation
    '''
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    label_loss_meter = utils.AvgrageMeter()
    domain_loss_meter = utils.AvgrageMeter() 
    model.eval()
    # alpha doesn't matter here since no backprop
    alpha = 0
    N = min( len( src_val_queue ), len( tgt_val_queue ) )
    src_val_queue_iter = iter( src_val_queue )
    tgt_val_queue_iter = iter( tgt_val_queue )
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        for step in range( N ):
            # get validation data
            src_images, src_labels = next( src_val_queue_iter )
            tgt_images, _ = next( tgt_val_queue_iter )
            src_images, src_labels, tgt_images = src_images.to( device ), \
                    src_labels.to( device ), tgt_images.to( device )
            batch_size = len( src_labels )
        
            # feed model using src_data
            src_domain = torch.zeros( batch_size ).long().to( device )
            src_labels_out, src_domain_out = model( src_images, alpha )
            src_label_loss = label_criterion( src_labels_out, src_labels )
            src_domain_loss = domain_criterion( src_domain_out, src_domain )

            # feed model using tgt_data
            tgt_domain = torch.ones( batch_size ).long().to( device )
            _, tgt_domain_out = model( tgt_images, alpha )
            tgt_domain_loss = domain_criterion( tgt_domain_out, tgt_domain )
       
            # calculate loss
            domain_loss = src_domain_loss + tgt_domain_loss
            loss = src_label_loss + domain_loss
        
            # calculate accuracy on src labels
            acc1, acc5 = utils.accuracy( src_labels_out, src_labels, topk=(1, 5) )

            # update stats
            objs.update( loss.item(), batch_size )
            top1.update( acc1.item(), batch_size )
            top5.update( acc5.item(), batch_size )
            label_loss_meter.update( src_label_loss.item(), batch_size )
            domain_loss_meter.update( domain_loss.item(), batch_size )

            # report stats
            if step % args.report_freq == 0:
                logging.info( 'train %03d %e %e %e %f %f', step, objs.avg,
                        label_loss_meter.avg, domain_loss_meter.avg,
                        top1.avg, top5.avg )
    
    return top1.avg, objs.avg            


if __name__ == '__main__':
    main()

