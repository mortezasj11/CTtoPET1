import argparse
import logging
import os

import numpy as np
import torch
#import torch.nn.functional as F
#from PIL import Image
#from torchvision import transforms

from unet import UNet
#from utils.data_vis import plot_img_and_mask
from utils.dataset_CTtoPET import Dataset_CTtoPET
import nibabel as nib


def predict_img(net, full_img, device):
    net.eval()
    img = torch.from_numpy(Dataset_CTtoPET.preprocessCT(full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img) 
    return output.squeeze().cpu().numpy()


#def get_output_filenames(args):
    #in_files = os.listdir(args.inputDir) 
    #out_files = []
    #for f in in_files:
        #pathsplit = f.split('.')
        #out_files.append("{}_OUT.nii.gz".format(pathsplit[0]))
    #return out_files


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default = 'C:\\Users\\MSalehjahromi\\Codes\\CT_to_PET\\checkpoints\\CP_epoch5.pth', metavar='FILE', help="Specify the file in which the model is stored")
    
    parser.add_argument('--inputDir', '-i', metavar='INPUT', nargs='+',
help='filenames of input images',default='C:/Users/MSalehjahromi/Data_ICON/CTPET_Train_Test_9thMarth/CT1_Ts/')
    
    parser.add_argument('--outputDir', '-o', 
help='filenames of output images',default='C:/Users/MSalehjahromi/Data_ICON/CTPET_Train_Test_9thMarth/CT1_Ts_predict/') 
    
    parser.add_argument('--saveDir', '-s', help="the output directory", default='C:/Users/MSalehjahromi/Data_ICON/CTPET_Train_Test/CT_Ts')
                              
    parser.add_argument('--save', '-sf', help="save the output masks", default=False)
    
    parser.add_argument('--isNIFTI', '-nf', help="is the iput a NIFTI file", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    in_files = os.listdir(args.inputDir) 
    #out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    for i, fn in enumerate(in_files[3:10]):
        logging.info("\nPredicting image {} ...".format(fn))

        full_fn = os.path.join(args.inputDir,fn)
        whole_img = nib.load(full_fn)
        img = whole_img.get_fdata()
        
    
        PET = np.zeros(img.shape)
            #print('Channel out of', img.shape[0]),
        
        # Going over all slices
        print(i, ') Predicting PET of   ', fn)
        for j in range(img.shape[2]):
            #print(j,end='')
            PET[:,:,j] = predict_img(net=net,
                           full_img=img[:,:,j],                         
                           device=device)
            
        #converting dta from float32 to uint16? maybe?
        
        
           
        # Saving to NIFTI        
        img_nifti = nib.Nifti1Image(PET*2000, whole_img.affine )
        
        pathsplit = fn.split('.')
        out_files = ("{}_OUT.nii.gz".format(pathsplit[0]))   
        img_nifti.to_filename(os.path.join(args.outputDir,out_files))
        
        print('Saved!:' ,out_files)
        print()
      
        
      
    '''
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        full_fn = os.path.join(args.inputDir,fn)
        img = np.load(full_fn)
               
        #if 3D data
        if len(img.shape)==3:
            PET = np.zeros(img.shape)
            print('Channel out of', img.shape[0]),
            for j in range(img.shape[0]):
                print(j),
                PET[j,:,:] = predict_img(net=net,
                               full_img=img[j,:,:],                         
                               device=device)
        #if 2D data                 
        elif len(img.shape)==2:
            PET = predict_img(net=net,
                               full_img=img,                         
                               device=device)
        #saving
        if args.save:
            out_fn = out_files[i]
            
            np.save( os.path.join(args.inputDir,out_files[i]), PET)
            logging.info("PET saved to {}".format(os.path.join(args.inputDir,out_files[i])))
            
        if args.isNIFTI:
            img_nifti = nib.Nifti1Image(PET, np.eye(4 ) )
            img_nifti.to_filename(os.path.join(args.inputDir, out_files[i].split('.')[0]+'.nii.gz'))
    '''























