import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def GiveImageAndTargetLists(main_path):
    images_list = []
    target_list = []
    for folder_name in os.listdir(main_path):
        whole_path = main_path + folder_name +"/"+ folder_name+"/"
        

        if os.path.isdir(whole_path):
            print(whole_path)
            images_path  = whole_path + "data/"
            targets_path = whole_path + "Lung/"
            
            len_imgs = len([images_path+img for img in os.listdir(images_path) if img.split(".")[-1]=='mat'])
            len_trgs = len([targets_path+trg for trg in os.listdir(targets_path) if trg.split(".")[-1]=='mat'])
            if len_imgs == len_trgs:
                images_list.append( [ images_path+img for img in os.listdir(images_path)  if img.split(".")[-1]=='mat'])
                target_list.append( [targets_path+trg for trg in os.listdir(targets_path) if trg.split(".")[-1]=='mat'])
            else:

                imgs_list =[  images_path+img for img in os.listdir(images_path)  if img.split(".")[-1]=='mat' and (img.split(".")[0]+'-lung.mat')   in os.listdir(targets_path)]
                trg_list = [ targets_path+trg for trg in os.listdir(targets_path) if trg.split(".")[-1]=='mat' and (trg.split(".")[0][:-5] + '.mat') in os.listdir(images_path) ]
                images_list.append(imgs_list)
                target_list.append(trg_list)
                
    return (images_list,target_list)







if __name__=='__main__':
    #mat = scipy.io.loadmat('file.mat')
    main_path = "C:/Users/MSalehjahromi/Downloads/"
    
    images_list,target_list = GiveImageAndTargetLists(main_path)            
    print(len(images_list),len(target_list))
    for i in range(len(images_list)):
        print(len(images_list[i]))