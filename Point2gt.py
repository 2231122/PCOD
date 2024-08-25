
import os
import cv2
import numpy as np
import json


Pred_Map_path='./CAMO'
GT_path='./GT'
Cu='./Cu'

save_path='./Final_GT'
file_list=sorted(os.listdir(Pred_Map_path))

for file in file_list:
        # #GT=cv2.imread(os.path.join(CAMO_path,file),cv2.IMREAD_GRAYSCALE)
        GT_ = cv2.imread(os.path.join(GT_path, file), cv2.IMREAD_GRAYSCALE)
        # GT_[GT == 1] = 100
        # GT_[GT == 2] = 255
        mask_orig=cv2.imread(os.path.join(Pred_Map_path,file),cv2.IMREAD_GRAYSCALE)

        init_cu=cv2.imread(os.path.join(Cu,file),cv2.IMREAD_GRAYSCALE)

        W,H=init_cu.shape
        #
        bg=init_cu.copy()
        fg=init_cu.copy()
        #
        bg[bg == 1] = 0

        fg[fg == 2] = 0
        fg[fg == 1] =150

        P = bg.argmax()
        center_x=P%H + 5
        center_y=P//H + 5


        # GT does not provide pixel information, only templates !!!
        GT_=GT_*0


        mask_orig[mask_orig <= 150] = 0
        mask_orig[mask_orig>150]=1
        #a是超参数！！！[2,4,6,8,16]
        a=2
        #mun=sum(sum(fg))
        #fg[cu == 1] = 150

        b_fg=fg.copy()
        num_c=0

        while np.max(b_fg)>105:
                P = b_fg.argmax()
                fg_x = P % H
                fg_y = P // H
                b_fg[fg_y:fg_y+11, fg_x:fg_x+11] = 0
                b_fg[fg_y+5,fg_x+5]=100
                num_c=num_c+1
        fg=b_fg
        mun=num_c
        count=sum(sum(mask_orig))//(mun*a)
        count=int(pow(count,0.5))
        count=int(count//2)

        for i in range(mun):
                P = fg.argmax()
                fg_x=P%H
                fg_y=P//H
                fg[fg_y,fg_x]=0
                GT_ = cv2.circle(GT_, center=(fg_x, fg_y), radius=count, color=(1, 1, 1),
                                                      thickness=-1)  # seed_point: (column, row)
        GT_ = cv2.circle(GT_, center=(center_x, center_y), radius=count, color=(2, 1, 1),
                         thickness=-1)
        #GT_[cu==2]=2
        cv2.imwrite(os.path.join(save_path, file), GT_)
print('Finished EdgePoint2gt.py')
