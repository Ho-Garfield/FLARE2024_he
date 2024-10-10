import ants
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy as np
import scipy.ndimage as ndi
from skimage import filters
tmp_dir = r"./temp1"

img_sub = "_0000.nii.gz"
lab_sub = ".nii.gz"

subs ={"pre":"_C-pre_0000.nii.gz",
       "t2wi":"_T2WI_0000.nii.gz",
       "dwi":"_DWI_0000.nii.gz",
       "v":"_C--V_0000.nii.gz",
       "delay":"_C--Delay_0000.nii.gz",
       "a":"_C--A_0000.nii.gz",
       "inphase":"_InPhase_0000.nii.gz",
       "outphase":"_OutPhase_0000.nii.gz"

       }


if(not os.path.exists(tmp_dir)):
    os.makedirs(tmp_dir,exist_ok=True)
def clean_tmp(contain_str:str, tmp_dir = r"./temp")->None:
    for name in os.listdir(tmp_dir) :
        if contain_str in name:
            os.remove(os.path.join(tmp_dir, name) )


def max_region(_region):
    log = ""
    _region = _region.astype(np.uint8)
    if(np.all(_region == 0)):
        return _region
    # 标记连通域
    labeled_image, num_features = ndi.label(_region)

    # 计算各个连通域的大小
    sizes = ndi.sum(_region, labeled_image, range(num_features + 1))
    sorted_indices = np.argsort(sizes[1:])[::-1]        
    top1_size = sizes[sorted_indices[0]+1]
    # 找到最大的连通域
    max_label = sorted_indices[0] + 1  # 从1开始是因为0是背景标签
    log = log + f"\n\tregion_1_size:{top1_size}"
    max_region = np.zeros_like(_region)
    max_region[labeled_image == max_label] = 1
    return max_region

def body_registration_a_v(mri_pre_path,mri_a_path, out_dir):
    
    mri_pre_ants = ants.image_read(mri_pre_path)
    mri_a_ants = ants.image_read(mri_a_path)
    
    mri_pre_np = mri_pre_ants.numpy()
    p_80 = np.percentile(mri_pre_np,80)
    t = filters.threshold_otsu(mri_pre_np[mri_pre_np<p_80])
   
    mask_pre = mri_pre_np>t
    mask_pre_ants = ants.from_numpy(mask_pre.astype(np.float32),mri_pre_ants.origin,mri_pre_ants.spacing,mri_pre_ants.direction)

    mri_a_np = mri_a_ants.numpy()
    p_80 = np.percentile(mri_a_np,80)
    t = filters.threshold_otsu(mri_a_np[mri_a_np<p_80])
    
    mask_a = mri_a_np>t
    mask_a_ants = ants.from_numpy(mask_a.astype(np.float32),mri_a_ants.origin,mri_a_ants.spacing,mri_a_ants.direction)
    name = os.path.basename(mri_pre_path)
    reg = ants.registration(fixed=mask_pre_ants,moving=mask_a_ants,type_of_transform="Rigid",outprefix = os.path.join(tmp_dir, name))
    mri_a_ants = ants.apply_transforms(fixed=mask_pre_ants, moving=mri_a_ants, transformlist=reg['fwdtransforms'])
    mask_a_ants = ants.apply_transforms(fixed=mask_pre_ants, moving=mask_a_ants, transformlist=reg['fwdtransforms'],
                                              interpolator="nearestNeighbor")
    clean_tmp(name, tmp_dir=tmp_dir)
    ants.image_write(mri_a_ants,os.path.join(out_dir,os.path.basename(mri_a_path)))
    ants.image_write(mask_a_ants,os.path.join(out_dir,"mask",
                                                    str.replace(os.path.basename(mri_a_path),img_sub,lab_sub)
                                                    ))


def body_registration_phase(mri_pre_path,mri_inphase_path,mri_outphase_path,out_dir):
    
    mri_pre_ants = ants.image_read(mri_pre_path)
    mri_inphase_ants = ants.image_read(mri_inphase_path)
    mri_outphase_ants = ants.image_read(mri_outphase_path)
    
    mri_pre_np = mri_pre_ants.numpy()
    p_80 = np.percentile(mri_pre_np,80)
    t = filters.threshold_otsu(mri_pre_np[mri_pre_np<p_80])
   
    mask_pre = mri_pre_np>t
    mask_pre_ants = ants.from_numpy(mask_pre.astype(np.float32),mri_pre_ants.origin,mri_pre_ants.spacing,mri_pre_ants.direction)
    # ants.image_write(mask_pre_ants,"pre.nii.gz")

    mri_inphase_np = mri_inphase_ants.numpy()
    p_80 = np.percentile(mri_inphase_np,80)
    t = filters.threshold_otsu(mri_inphase_np[mri_inphase_np<p_80])

    mask_inphase = mri_inphase_np>t
    mask_inphase_ants = ants.from_numpy(mask_inphase.astype(np.float32),mri_inphase_ants.origin,mri_inphase_ants.spacing,mri_inphase_ants.direction)
    # ants.image_write(mask_inphase_ants,"IN.nii.gz")

    name = os.path.basename(mri_pre_path)
    reg = ants.registration(fixed=mask_pre_ants,moving=mask_inphase_ants,type_of_transform="Rigid",outprefix = os.path.join(tmp_dir, name))
    mri_inphase_ants = ants.apply_transforms(fixed=mask_pre_ants, moving=mri_inphase_ants, transformlist=reg['fwdtransforms'])
    mri_outphase_ants = ants.apply_transforms(fixed=mask_pre_ants, moving=mri_outphase_ants, transformlist=reg['fwdtransforms'])
    mask_inphase_ants = ants.apply_transforms(fixed=mask_pre_ants, moving=mask_inphase_ants, transformlist=reg['fwdtransforms'],
                                              interpolator="nearestNeighbor")
    clean_tmp(name, tmp_dir=tmp_dir)
    ants.image_write(mri_inphase_ants,os.path.join(out_dir,os.path.basename(mri_inphase_path)))
    ants.image_write(mri_outphase_ants,os.path.join(out_dir,os.path.basename(mri_outphase_path)))
    ants.image_write(mask_inphase_ants,os.path.join(out_dir,"mask",
                                                    str.replace(os.path.basename(mri_inphase_path),img_sub,lab_sub)
                                                    ))

def body_registration_delay(mri_pre_path,mri_d_path, out_dir):
    
    mri_pre_ants = ants.image_read(mri_pre_path)
    mri_d_ants = ants.image_read(mri_d_path)
    
    mri_pre_np = mri_pre_ants.numpy()
    p_80 = np.percentile(mri_pre_np,80)
    t = filters.threshold_otsu(mri_pre_np[mri_pre_np<p_80])
   
    mask_pre = mri_pre_np>t
    mask_pre_ants = ants.from_numpy(mask_pre.astype(np.float32),mri_pre_ants.origin,mri_pre_ants.spacing,mri_pre_ants.direction)

    mri_d_np = mri_d_ants.numpy()
    t = filters.threshold_otsu(mri_d_np)
    
    mask_a = mri_d_np>t
    mask_d_ants = ants.from_numpy(mask_a.astype(np.float32),mri_d_ants.origin,mri_d_ants.spacing,mri_d_ants.direction)

    name = os.path.basename(mri_pre_path)
    reg = ants.registration(fixed=mask_pre_ants,moving=mask_d_ants,type_of_transform="Rigid",outprefix = os.path.join(tmp_dir, name))

    mri_d_ants = ants.apply_transforms(fixed=mask_pre_ants, moving=mri_d_ants, transformlist=reg['fwdtransforms'])
    mask_d_ants = ants.apply_transforms(fixed=mask_pre_ants, moving=mask_d_ants, transformlist=reg['fwdtransforms'],
                                              interpolator="nearestNeighbor")
    clean_tmp(name, tmp_dir=tmp_dir)
    ants.image_write(mri_d_ants,os.path.join(out_dir,os.path.basename(mri_d_path)))
    ants.image_write(mask_d_ants,os.path.join(out_dir,"mask",
                                                    str.replace(os.path.basename(mri_d_path),img_sub,lab_sub)
                                                    ))

def registration_wi2a(mri_a_path,mri_wi_path, out_dir):
    
    mri_a_ants = ants.image_read(mri_a_path)
    mri_wi_ants = ants.image_read(mri_wi_path)
    
    name = os.path.basename(mri_a_path)
    reg = ants.registration(fixed=mri_a_ants,moving=mri_wi_ants,type_of_transform="Rigid",outprefix = os.path.join(tmp_dir, name))
    mri_wi_ants = ants.apply_transforms(fixed=mri_a_ants, moving=mri_wi_ants, transformlist=reg['fwdtransforms'])
    clean_tmp(name, tmp_dir=tmp_dir)
    ants.image_write(mri_wi_ants,os.path.join(out_dir,os.path.basename(mri_wi_path)))






if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--in_dir', type=str, default='./STD/MRI/LLD')
    parser.add_argument('--out_dir', type=str, default='./data/stage2/images')    
    parser.add_argument('--num_process', type=int, default=5)
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir

    if not os.path.exists(os.path.join(out_dir,"mask")):
        os.makedirs(os.path.join(out_dir,"mask"))
    mri_pre_paths = [os.path.join(in_dir,f) for f in os.listdir(in_dir) if f.endswith(subs['pre']) ]
    mri_v_paths = [str.replace(f,subs["pre"],subs["v"]) for f in mri_pre_paths]
    mri_a_paths = [str.replace(f,subs["pre"],subs["a"]) for f in mri_pre_paths]
    mri_d_paths = [str.replace(f,subs["pre"],subs["delay"]) for f in mri_pre_paths]
    mri_inphase_paths = [str.replace(f,subs["pre"],subs["inphase"]) for f in mri_pre_paths]
    mri_outphase_paths = [str.replace(f,subs["pre"],subs["outphase"]) for f in mri_pre_paths]

    import shutil
    for img_path in mri_pre_paths:
        out_path = os.path.join(args.out_dir,os.path.basename(img_path))
        if not os.path.exists(out_path):
            shutil.copy2(img_path,out_path)



    ptqdm(function = body_registration_a_v, iterable = (mri_pre_paths,mri_a_paths), processes = args.num_process,
        zipped=True,out_dir=out_dir
    )  
    ptqdm(function = body_registration_a_v, iterable = (mri_pre_paths,mri_v_paths), processes = args.num_process,
        zipped=True,out_dir=out_dir
    )  
    ptqdm(function = body_registration_delay, iterable = (mri_pre_paths,mri_d_paths), processes = args.num_process,
    zipped=True,out_dir=out_dir
    )

    mri_ra_paths = [os.path.join(out_dir,f) for f in os.listdir(out_dir) if f.endswith(subs['a']) ]
    mri_t2wi_paths = [str.replace(str.replace(f,out_dir,in_dir),subs["a"],subs["t2wi"]) for f in mri_ra_paths]
    mri_dwi_paths = [str.replace(str.replace(f,out_dir,in_dir),subs["a"],subs["dwi"]) for f in mri_ra_paths]

    ptqdm(function = registration_wi2a, iterable = (mri_ra_paths,mri_t2wi_paths), processes = args.num_process,
        zipped=True,out_dir=out_dir
    )  
    ptqdm(function = registration_wi2a, iterable = (mri_ra_paths,mri_dwi_paths), processes = args.num_process,
        zipped=True,out_dir=out_dir
    )  
    ptqdm(function = body_registration_phase, iterable = (mri_pre_paths,mri_inphase_paths,mri_outphase_paths), processes = args.num_process,
        zipped=True,out_dir=out_dir
    )      


