import os
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
from skimage import filters
from acvl_utils.miscellaneous.ptqdm import ptqdm
import shutil
subs ={"pre":"_C-pre_0000.nii.gz",
       "t2wi":"_T2WI_0000.nii.gz",
       "dwi":"_DWI_0000.nii.gz",
       "v":"_C--V_0000.nii.gz",
       "delay":"_C--Delay_0000.nii.gz",
       "a":"_C--A_0000.nii.gz",
       "inphase":"_InPhase_0000.nii.gz",
       "outphase":"_OutPhase_0000.nii.gz",
       "label":".nii.gz",
       "image":"_0000.nii.gz",
       "pre_label":"_C-pre.nii.gz"
       }
LABELS= {
        "background": 0,
        "Liver": 1,
        "Right Kidney": 2,
        "Spleen": 3,
        "Pancreas": 4,
        "Aorta": 5,
        "Inferior Vena Cava": 6,
        "Right Adrenal Gland": 7,
        "Left Adrenal Gland": 8,
        "Gall Bladder": 9,
        "Esophagus": 10,
        "Stomach": 11,
        "Duodenum": 12,
        "Left Kidney": 13
        }
def max_region(_region,box=None):
    log = ""
    _region = _region.astype(np.uint8)
    if(np.all(_region == 0)):
        return _region
    # 标记连通域
    labeled_image, num_features = ndi.label(_region)
    if box is None:
        # 计算各个连通域的大小
        sizes = ndi.sum(_region, labeled_image, range(num_features + 1))
    else:
        sizes = ndi.sum(_region, labeled_image*box, range(num_features + 1))
    sorted_indices = np.argsort(sizes[1:])[::-1]        
    top1_size = sizes[sorted_indices[0]+1]
    # 找到最大的连通域
    max_label = sorted_indices[0] + 1  # 从1开始是因为0是背景标签
    log = log + f"\n\tregion_1_size:{top1_size}"
    max_region = np.zeros_like(_region)
    max_region[labeled_image == max_label] = 1
    return max_region

def check_match(check_mask_path, mris_dir, match_dir):
    name = str.replace(os.path.basename(check_mask_path),subs['pre_label'],"")
    rough_mask_sitk = sitk.ReadImage(check_mask_path)
    rough_mask = sitk.GetArrayFromImage(rough_mask_sitk)
    ks = subs.keys() - ['pre','outphase','label','image',"pre_label"]
    kid_pre = rough_mask
    label_mistake = True
    for k in ks:
        mismatch_side = 0
        mri_path = os.path.join(mris_dir,name+subs[k])
        mri = sitk.ReadImage(mri_path)
        kid_k = get_kidney_label(rough_mask,sitk.GetArrayFromImage(mri),k)
        for lab, i in zip([LABELS["Left Kidney"],LABELS["Right Kidney"]],range(0,2)):
            cur_label = kid_pre ==lab
            match_label = (kid_k==lab)
            # 计算 Dice 系数
            intersection = np.sum(cur_label * match_label)
            union = np.sum(cur_label) + np.sum(match_label)
            dice = (2.0 * intersection+0.001) / (union+0.001)
            print(f"match {name}: (pre, {k}), dice{dice}")
            if k=="dwi" or k=="t2wi" or k =="inphase":
                if dice < 0.8:
                    mismatch_side = 2
                    break

            if dice < 0.85:
                mismatch_side = mismatch_side + 1
        if mismatch_side ==2:
            kindney_label_sitk = sitk.GetImageFromArray(kid_k)
            kindney_label_sitk.CopyInformation(rough_mask_sitk)
            print(f"mismatch {name}: (pre, {k}), dice{dice}")
        else:
            label_mistake = False
            match_path  = os.path.join(match_dir,"images",os.path.basename(mri_path))
            sitk.WriteImage(mri,match_path)
            if k == "inphase":
                outphase_name = str.replace(os.path.basename(match_path),subs["inphase"],subs["outphase"])
                outphase_path = os.path.join(mris_dir,outphase_name)
                match_path = str.replace(match_path,subs["inphase"],subs["outphase"])
                shutil.copy(outphase_path,match_path)
    if label_mistake:
        return

    mri_pre_path  = os.path.join(mris_dir,name+subs['pre'])
    match_path = os.path.join(match_dir,"images",name+subs['pre'])
    shutil.copy(mri_pre_path,match_path)
    kindney_label_sitk = sitk.GetImageFromArray(rough_mask)
    kindney_label_sitk.CopyInformation(rough_mask_sitk)
    match_mask_path = os.path.join(match_dir, "masks", name+subs['label'])
    sitk.WriteImage(kindney_label_sitk,match_mask_path)

def get_kidney_label(rough_label,mri,k):
    kidney_label = np.zeros(rough_label.shape) 
    mask_avoid =(rough_label!=LABELS['Liver'])*(rough_label!=LABELS["Spleen"])\
        *(rough_label!=LABELS["Duodenum"])*(rough_label!=LABELS["Right Adrenal Gland"])*(rough_label!=LABELS["Left Adrenal Gland"])
    for label in [LABELS["Right Kidney"],LABELS["Left Kidney"]]: 
        kidney = rough_label == label
        expand_kidney = ndi.binary_dilation(kidney,iterations=6)*mask_avoid
        if np.sum(expand_kidney) == 0:
            continue
        v = filters.threshold_otsu(mri[expand_kidney])
        if k == 'inphase':   
            kidney = (mri < v)*expand_kidney*(mri>0)
            kidney = ndi.binary_dilation(ndi.binary_erosion(kidney,iterations=2),iterations=2)
        else:
            kidney = (mri > v)*expand_kidney

        if k == 'a':
            kidney = ndi.binary_dilation(max_region(ndi.binary_erosion(kidney,iterations=1)),iterations=1)
        else:
            zs,ys,xs = np.where(kidney==1)
            for z in np.unique(zs):
                kidney[z] = ndi.binary_dilation(max_region(ndi.binary_erosion(kidney[z],iterations=1)),iterations=1)
        if np.sum(kidney) == 0:
            continue    
        expand_kidney = ndi.binary_dilation(kidney,iterations=2)

        v = np.mean(mri[expand_kidney])            
        if k == "inphase": 
            kidney = ((mri<v)*expand_kidney*(mri>0))|kidney     
        else:
            kidney = ((mri>v)*expand_kidney)|kidney
        
        if k =="a":
            e = 4
            expand = np.zeros([s + 2*e for s in kidney.shape])
            expand[e:-e,e:-e,e:-e] = kidney
            kidney = ndi.binary_erosion(ndi.binary_dilation(expand,iterations=e),iterations=e)[e:-e,e:-e,e:-e]
        else :
            kidney = ndi.binary_dilation(ndi.binary_erosion(kidney,iterations=1),iterations=1)
            e = 2
            expand = np.zeros([s + 2*e for s in kidney.shape])
            expand[e:-e,e:-e,e:-e] = kidney
            kidney = ndi.binary_erosion(ndi.binary_dilation(expand,iterations=e),iterations=e)[e:-e,e:-e,e:-e]
        kidney= kidney*mask_avoid
        kidney = max_region(kidney)
        kidney_label[kidney==1]=label
    return kidney_label
import shutil
def mask2image(rough_label_path, mask_mris_dir, out_dir):
    name = str.replace(os.path.basename(rough_label_path),subs['label'],"")
    for k in subs.keys():
        mri_path = os.path.join(mask_mris_dir,name+subs[k])
        mri_mask_name = str.replace(os.path.basename(mri_path),subs["image"],subs["label"])
        mri_mask_path = os.path.join(out_dir,mri_mask_name)
        if os.path.exists(mri_path):
            shutil.copy(rough_label_path,mri_mask_path)




if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--mris_dir', type=str, default='data/stage2/images')
    parser.add_argument('--mask_dir', type=str, default='./data/stage1_pred_t1w') 
    parser.add_argument('--temp_dir', type=str, default='./data/stage2_temp')
    parser.add_argument('--final_label_dir', type=str, default='data/stage2/labels')
    parser.add_argument('--num_process', type=int, default=8)
    args = parser.parse_args()
    mris_dir = args.mris_dir
    mask_dir = args.mask_dir
    match_dir = args.temp_dir
    final_label_dir = args.final_label_dir

    if not os.path.exists(os.path.join(match_dir,"masks")):
        os.makedirs(os.path.join(match_dir,"masks"))
    if not os.path.exists(os.path.join(match_dir,"images")):
        os.makedirs(os.path.join(match_dir,"images")) 
    if not os.path.exists(final_label_dir):
        os.makedirs(final_label_dir)
    label_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(subs['pre_label']) ]
    ptqdm(function = check_match, iterable = label_paths, processes = args.num_process, 
          mris_dir=mris_dir, match_dir=match_dir)  
    labeld_pa_path = [os.path.join(os.path.join(match_dir,"masks"), f) for f in os.listdir(os.path.join(match_dir,"masks")) if f.endswith(subs['label']) ]
    ptqdm(function = mask2image, iterable = labeld_pa_path, processes = args.num_process,
        mask_mris_dir=os.path.join(match_dir,"images"),out_dir=final_label_dir
    )
    shutil.rmtree(args.temp_dir)

