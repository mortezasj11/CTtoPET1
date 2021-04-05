import os
import scipy.io
import numpy as np
import nibabel as nib
import shutil
import SimpleITK as sitk
import skimage.io as io
import shutil


def GiveImageAndTargetLists(main_path):
    #C:/Users/MSalehjahromi/Data_ICON/
    CT_list = []
    PET_list = []

    CT_path_file  = main_path + "CT_mhd_mori/"
    PET_path_file = main_path + "PET_reg_mhd_Sheeba/"

    for folder_name in os.listdir(CT_path_file):
        #print(folder_name)
        whole_path_CT = CT_path_file + folder_name +"/"
        whole_path_PET = PET_path_file + folder_name +"/"

        if os.path.isdir(whole_path_CT) & os.path.isdir(whole_path_PET):
            #print(whole_path_CT)
            CT_path  = whole_path_CT + "CT_conv.mhd"
            PET_path = whole_path_PET + "result.0.mhd"

            CT_list.append(CT_path)
            PET_list.append(PET_path)
            
    return (CT_list, PET_list)


def SavingAsNpy(CT_list, PET_list, CT_Tr_path, PET_Tr_path, CT_Ts_path, PET_Ts_path, prefix=""):

    count_ts = 0
    count_tr = 0
    for j in range(37,len(CT_list)):
        print(j,'/',len(CT_list)-1, end=' ')
        #print()

        if PET_list[0].split('/')[-2] == CT_list[0].split('/')[-2]:
           
            #Save ~30% of patients data in the test files
            if j < len(CT_list)*0.3:
                         
                CT = sitk.ReadImage(CT_list[j])
                dst_CT_name = "CTPET_3D_"+ prefix +'_'+ str(count_ts).zfill(6) + ".nii.gz"
                dst_CT_path = os.path.join(CT_Ts_path, dst_CT_name)
                #sitk.WriteImage(CT,dst_CT_path )
                
                PET = sitk.ReadImage(PET_list[j])
                dst_PET_name = "CTPET_3D_"+ prefix +'_'+ str(count_ts).zfill(6) + ".nii.gz"
                dst_PET_path = os.path.join(PET_Ts_path, dst_PET_name)
                #sitk.WriteImage(PET,dst_PET_path )
                               
                count_ts += 1
            

            else:
                CT = io.imread(CT_list[j], plugin='simpleitk') #CT = np.array(CT)   
                PET = io.imread(PET_list[j], plugin='simpleitk') #PET = np.array(PET)
                #print(PET.shape)
                #print(PET.shape[1:3])
                #print(PET.shape[])
                if PET.shape[1:3] != (512,512):
                    print(PET.shape)
                    print(PET_list[j])
                    print(PET.shape)
                
                CT_path  = CT_Tr_path
                PET_path   = PET_Tr_path
                
                '''
                #Saving channel CT & PET images 
                for k in range(CT.shape[0]):

                    CT_k = np.array( CT[k,:,:] )
                    PET_k = np.array( PET[k,:,:] )

                    dst_img_name = "CTPET_"+ prefix +'_'+ str(count_tr).zfill(6) + ".npy"
                    dst_img_path = os.path.join(CT_path, dst_img_name)
                    np.save(dst_img_path, CT_k)

                    dst_label_name = "CTPET_"+ prefix +'_'+ str(count_tr).zfill(6) + ".npy"
                    dst_mask_path = os.path.join(PET_path, dst_label_name)
                    np.save(dst_mask_path, PET_k)
                
                    count_tr += 1
                '''
    return (count_ts, count_tr)


if __name__=='__main__':

    # Destination directory
    main_folder = '/Data/CTtoPET/CTPET_Train_Test_Marth16'
    os.makedirs(main_folder,exist_ok=True)
    #if os.path.exists(nnUnet_im_lbl_folder):
        #shutil.rmtree(nnUnet_im_lbl_folder)

    CT_Tr_path = os.path.join(main_folder, "CT1_Tr")
    os.makedirs(CT_Tr_path,exist_ok=True)

    PET_Tr_path = os.path.join(main_folder, "PET1_Tr")
    os.makedirs(PET_Tr_path,exist_ok=True)

    CT_Ts_path = os.path.join(main_folder, "CT1_Ts")
    os.makedirs(CT_Ts_path,exist_ok=True)

    PET_Ts_path = os.path.join(main_folder, "PET1_Ts")
    os.makedirs(PET_Ts_path,exist_ok=True)
    
    #Getting images_list & target_list
    raw_dot_m_files = '/Data/CTtoPET/MainRaw/'
    CT_list, PET_list = GiveImageAndTargetLists(raw_dot_m_files)
    print("len(CT_list) & len(PET_list):",len(CT_list),'  &  ' ,len(PET_list))

    prefix = ""

    #SavingD
    SavingAsNpy(CT_list,    PET_list,  
                CT_Tr_path,  PET_Tr_path, 
                CT_Ts_path,  PET_Ts_path, 
                prefix=prefix)