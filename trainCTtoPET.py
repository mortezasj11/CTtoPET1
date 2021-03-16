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
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset_CTtoPET import Dataset_CTtoPET
from torch.utils.data import DataLoader, random_split
#19,20: dir ,157Channel,33,16:BasicDataset_CTtoPET,
dir_CT = 'C:/Users/MSalehjahromi/Data_ICON/CTPET_Train_Test_9thMarth/CT1_Tr'
dir_PET = 'C:/Users/MSalehjahromi/Data_ICON/CTPET_Train_Test_9thMarth/PET1_Tr'

today = datetime.now()
dir_checkpoint = os.getcwd()+'\\checkpoints\\'+ today.strftime('%Y%m%d_%H%M\\')

def train_net(net,
              device,
              epochs=1,
              batch_size=16,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              start_saving=0 ):

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
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    criterion = nn.MSELoss(reduction='mean')
    ma_loss = 0.0003
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                CTs = batch['CT'] #[2, 1, 512, 512]
                PETs = batch['PET']
                assert CTs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {CTs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                CTs = CTs.to(device=device, dtype=torch.float32)
                #mask_type = torch.float32 if net.n_classes == 1 else torch.long
                mask_type = torch.float32
                PETs = PETs.to(device=device, dtype=mask_type)

                PETs_pred = net(CTs)
                loss = criterion(PETs_pred, PETs)
                epoch_loss += loss.item()
                
                ma_loss = 0.999*ma_loss + 0.001*loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': ma_loss})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(CTs.shape[0])
                global_step += 1
                
        print('epoch_loss:', epoch_loss)
        print()

        #'''
        # Validation
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
        val_score = eval_net(net, val_loader, device)
        scheduler.step(val_score)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        logging.info('Validation mse_loss: {}'.format(val_score))
        writer.add_scalar('Loss/test', val_score, global_step)

        writer.add_images('images', CTs, global_step)

        writer.add_images('PETs/true', PETs, global_step)
        writer.add_images('PETs/pred', PETs_pred, global_step)
        #'''
                        
        
        if save_cp:
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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=3,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    
    #load_pth = 'C:\\Users\\MSalehjahromi\\Codes\\CT_to_PET\\checkpoints\\*\\CP_epoch22.pth'
    parser.add_argument('-f', '--load', dest='load',type=str,default=False , help='Load model from a .pth file')
    
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    
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
                  start_saving = int(start_saving))
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