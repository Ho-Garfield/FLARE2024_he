import SimpleITK as sitk
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy as np
from skimage import filters
import shutil
inphase_dir ="/DATA_16T/MICCAI/FLARE/FLARE24-Task3-MR/Training/LLD-MMRI-3984"
handle_dir = "/DATA_16T/MICCAI/FLARE/FLARE24-Task3-MR/Training/LLD-MMRI-3984"
out_dir = "/DATA_16T/MICCAI/Flip_resample"

subs =["_T2WI_0000.nii.gz","_OutPhase_0000.nii.gz","_DWI_0000.nii.gz","_C--V_0000.nii.gz",
       "_C-pre_0000.nii.gz","_C--Delay_0000.nii.gz","_C--A_0000.nii.gz","_InPhase_0000.nii.gz"]

def check(i,handle_dir):
    image= sitk.ReadImage(i)
    image_np = sitk.GetArrayViewFromImage(image)
    image_np = (image_np/image_np.max())*255
    t = filters.threshold_otsu(image_np)
    mask = image_np>t
    _max = mask[0].shape[0]*mask[0].shape[1]*9./10.
    _min = 100
    for j in range(0,10):
        if _max>np.sum(mask[j])>_min:
            mask_low = mask[j]
            break
    for j in range(mask.shape[0]-1,mask.shape[0] - 11,-1):
        if _max>np.sum(mask[j])>_min:
            mask_up = mask[j]
            break


    
    y,x = np.where(mask_low==1)
    box1 = np.zeros(mask_low.shape)
    box1[min(y):max(y),min(x):max(x)] = 1

    y,x = np.where(mask_up==1)
    box2 = np.zeros(mask_up.shape)
    box2[min(y):max(y),min(x):max(x)] = 1



    hole_low = np.sum((box1>0)*(mask_low==0))
    hole_up = np.sum((box2>0)*(mask_up==0))
    if (hole_low>hole_up):
        for sub in subs:
            in_path = os.path.join(handle_dir,os.path.basename(str.replace(i,"_InPhase_0000.nii.gz",sub)))
            img_sitk = sitk.ReadImage(in_path)
            img = sitk.GetArrayFromImage(img_sitk)
            flip_img = np.flip(img,axis=0)
            flip_img_sitk = sitk.GetImageFromArray(flip_img)
            flip_img_sitk.CopyInformation(img_sitk)
            out_path = os.path.join(out_dir,os.path.basename(str.replace(i,"_InPhase_0000.nii.gz",sub)))
            print(out_path)
            sitk.WriteImage(flip_img_sitk, out_path)        
    else:
        for sub in subs:
            in_path = os.path.join(handle_dir,os.path.basename(str.replace(i,"_InPhase_0000.nii.gz",sub)))
            out_path = os.path.join(out_dir,os.path.basename(str.replace(i,"_InPhase_0000.nii.gz",sub)))
            shutil.copy(in_path,out_path)





if __name__ == "__main__":
    img_paths = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".nii.gz") ]
    ptqdm(function = check, iterable = img_paths, processes = 4, handle_dir = handle_dir
        )

