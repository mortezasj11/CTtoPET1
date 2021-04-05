import numpy as np
import os

path = '/Data/CTtoPET/CTPET_Train_Test_Marth16/PET1_Tr'

if __name__ == '__main__':
    j=0
    i=0
    for file_name in os.listdir(path):
        #if i%100 == 0:
            #print(i,'/', len(os.listdir(path)))
        #print(os.path.join(path, file_name))

        full_file_path = os.path.join(path, file_name)
        img = np.load(full_file_path)
        #print(img.shape, end ='')
        
        if img.shape != (512,512):
            print(i)
            print(os.path.join(path, file_name))
            print()
            j+=1
        i += 1
    
    print('Number of cased',j)