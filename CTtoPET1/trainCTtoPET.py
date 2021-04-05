import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import *

from torch.utils.tensorboard import SummaryWriter
from utils.dataset_CTtoPET import Dataset_CTtoPET
from torch.utils.data import DataLoader, random_split

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#19,20: dir ,157Channel,33,16:BasicDataset_CTtoPET,

main_data_path = '/Data/CTtoPET/CTPET_Train_Test_Marth16/'
dir_CT = main_data_path + 'CT1_Tr'
dir_PET = main_data_path + 'PET1_Tr'

today = datetime.now()
dir_checkpoint = os.getcwd()+'/checkpoints/'+ today.strftime('%Y%m%d_%H%M/')

def train_net(net,
              device,
              epochs=1,
              batch_size=16,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              start_saving=0, reduction_loss = 'mean' ):

    dataset = Dataset_CTtoPET(dir_CT, dir_PET)
    n_val = int(len(dataset) * val_percent)   #2685
    n_train = len(dataset) - n_val  #24168
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        reduction_loss: {reduction_loss}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6)

    criterion1 = nn.MSELoss(reduction=reduction_loss)
    ma_loss = 0.0003
    for epoch in range(epochs):
        net.train()


        ############## Training ##############
        ############## Training ##############
        epoch_loss_sum = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                CTs = batch['CT'] #[16, 1, 512, 512]
                PETs = batch['PET']
                assert CTs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {CTs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                CTs = CTs.to(device=device, dtype=torch.float32)

                PET_type = torch.float32
                PETs = PETs.to(device=device, dtype=PET_type)

                PETs_pred = net(CTs)
                # Loss
                loss1 = criterion1(PETs_pred, PETs)/batch_size
                epoch_loss_sum += loss1.item()

                loss2 = 1 - ms_ssim( X, Y, data_range=1.0, size_average=True )
                
                ma_loss = 0.999*ma_loss + 0.001*loss1.item()
                loss= W1*loss1 + W2*loss2

                pbar.set_postfix(**{'loss (batch)': ma_loss})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(CTs.shape[0])
                global_step += 1
                
            writer.add_scalar('Loss/training', epoch_loss_sum/len(train_loader), global_step)
        print('epoch_loss:', epoch_loss_sum/len(train_loader))
        print()

        # Batch saving on training
            #writer.add_images('images', CTs, global_step)
            #writer.add_images('PETs/true_train', PETs, global_step)
            #writer.add_images('PETs/pred_train', PETs_pred, global_step)

        ############## Validation ##############
        ############## Validation ##############
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

        val_score, PET_true_val, PET_pred_val = eval_net(net, val_loader, device, reduction_loss)

        scheduler.step(val_score)
        
        # Batch saving on validation
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Loss/validation', val_score, global_step)
        writer.add_images('PETs/true_validation', PET_true_val, global_step)
        writer.add_images('PETs/pred_validation', PET_pred_val, global_step)
                        
        logging.info('Validation mse_loss: {}'.format(val_score))

        ############## Saving checkpoint ##############
        ############## Saving checkpoint ##############
        if save_cp and epoch%5==0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{start_saving + epoch + 1}.pth')
            logging.info(f'Checkpoint {start_saving + epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    
    #load_pth = 'C:\\Users\\MSalehjahromi\\Codes\\CT_to_PET\\checkpoints\\*\\CP_epoch22.pth'
    parser.add_argument('-f', '--load', dest='load',type=str,default=False , help='Load model from a .pth file')
    
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    parser.add_argument('-rl', '--reduction_loss', type=str, default= 'mean',
                        help=' mean or loss ', dest='reduction_loss')

    parser.add_argument('--loadUNet', '-lu', help='loading UNet or UNet_AU', default='UNet') 

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    logging.info(f'Using device {device}')

    #net = UNet(n_channels=1, n_classes=1, bilinear=True)
    if args.loadUNet =='UNet':
        net = UNet(n_channels=1, n_classes=1)
    elif args.loadUNet =='UNet_AU':
        net = UNet_AU(n_channels=1, n_classes=1)

    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    start_saving = 0
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')     
        start_saving = args.load.split('\\')[-1].split('.')[0][8:] #Based on the saving name CP_epoch{*}.pth

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  start_saving = int(start_saving),
                  reduction_loss = args.reduction_loss)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), dir_checkpoint + 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

'''
import torch
torch.cuda.empty_cache()
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0) 
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
'''



'''
For ssim, it is recommended to set nonnegative_ssim=True to avoid negative results. However, this option is set to False by default to keep it consistent with tensorflow and skimage.

# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  

# calculate ssim & ms-ssim for each image
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# reuse the gaussian kernel with SSIM & MS_SSIM. 
ssim_module = SSIM(data_range=255, size_average=True, channel=3)
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)
'''