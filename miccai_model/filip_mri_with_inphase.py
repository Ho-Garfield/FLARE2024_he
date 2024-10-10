import SimpleITK as sitk
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy as np
from skimage import filters
import shutil


def check(i,handle_dir,out_dir):
    image= sitk.ReadImage(i)
    image_np = sitk.GetArrayViewFromImage(image)
    image_np = (image_np/image_np.max())*255
    t = filters.threshold_otsu(image_np)
    mask = image_np>t
    _max = mask[0].shape[0]*mask[0].shape[1]*9./10.
    _min = 100
    mask_low = mask_up = None
    for j in range(0,10):
        if _max>np.sum(mask[j])>_min:
            mask_low = mask[j]
            break
    for j in range(mask.shape[0]-1,mask.shape[0] - 11,-1):
        if _max>np.sum(mask[j])>_min:
            mask_up = mask[j]
            break
    if mask_up is None :
        print(f"error up:{i}")

    if mask_low is None:
        print(f"error low:{i}")

    
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
            print("flip:"+out_path)
            sitk.WriteImage(flip_img_sitk, out_path)        
    else:
        for sub in subs:
            in_path = os.path.join(handle_dir,os.path.basename(str.replace(i,"_InPhase_0000.nii.gz",sub)))
            out_path = os.path.join(out_dir,os.path.basename(str.replace(i,"_InPhase_0000.nii.gz",sub)))
            if not os.path.exists(out_path):
                shutil.copy(in_path,out_path)






subs =["_T2WI_0000.nii.gz","_OutPhase_0000.nii.gz","_DWI_0000.nii.gz","_C--V_0000.nii.gz",
       "_C-pre_0000.nii.gz","_C--Delay_0000.nii.gz","_C--A_0000.nii.gz","_InPhase_0000.nii.gz"]

if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--inphase_dir', type=str, default='./STD/MRI/LLD')
    parser.add_argument('--handle_dir', type=str, default='./STD/MRI/LLD')

    parser.add_argument('--out_dir', type=str, default='./STD/MRI/LLD')    
    parser.add_argument('--num_process', type=int, default=4)
    # 继续添加其他参数...
    args = parser.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)

    img_paths = [os.path.join(args.inphase_dir, f) for f in os.listdir(args.inphase_dir) if f.endswith("_InPhase_0000.nii.gz") ]
    ptqdm(function = check, iterable = img_paths, processes = args.num_process, handle_dir = args.handle_dir,out_dir=args.out_dir
        )

