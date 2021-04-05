from os.path import splitext
from os import listdir
from os.path import join
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from skimage.restoration import denoise_tv_chambolle


class Dataset_CTtoPET(Dataset):
    def __init__(self, CT_dir, PET_dir):
        
        self.CT_dir = CT_dir
        self.PET_dir = PET_dir
        self.ids = [file for file in listdir(CT_dir)
                    if not file.startswith('.')]  
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        


    def __len__(self):
        return len(self.ids)



    @classmethod
    def preprocessCT(cls, im):
        img_np = np.array(im)
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)
        # HWC to CHW
        img_trans = img_np.transpose((2, 0, 1))
        img_trans = np.clip(img_trans,-1024.0,300.0)
        img_trans = (img_trans + 1024.0)/(1024+300)      
        return img_trans


    
    @classmethod
    def preprocessPET(cls, im):
        img_np = np.array(im)
        img_np = img_np/100.0
        img_np = np.clip(img_np,0.0,20.0)
        img_np = img_np/20.0
        #denoising
        img_np = denoise_tv_chambolle(img_np, weight=0.03)
        #add dimension
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)
        # HWC to CHW
        img_trans = img_np.transpose((2, 0, 1))

        return img_trans 
 
    
 

    def __getitem__(self, i):
        idx = self.ids[i]
        PET_file = join( self.PET_dir , idx )
        CT_file = join(self.CT_dir , idx )
                 
        PET = np.load(PET_file)
        CT = np.load(CT_file)

        CT = self.preprocessCT(CT)
        PET = self.preprocessPET(PET)

        return {
            'CT': torch.from_numpy(CT).type(torch.FloatTensor),
            'PET': torch.from_numpy(PET).type(torch.FloatTensor)
        }
