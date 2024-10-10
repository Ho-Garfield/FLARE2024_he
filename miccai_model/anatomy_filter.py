import os
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
from acvl_utils.miscellaneous.ptqdm import ptqdm
import shutil
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
       }

def is_region_only_one(_region, min_size_seen_as_region=64):
    _region = _region.astype(np.uint8)
    if(np.all(_region == 0)):
        return _region
    # 标记连通域
    labeled_image, num_features = ndi.label(_region)
    if num_features == 1:
        return True
    # 计算各个连通域的大小
    sizes = ndi.sum(_region, labeled_image, range(num_features + 1))
    sorted_indices = np.argsort(sizes[1:])[::-1]        
    top2_size = sizes[sorted_indices[1]+1]
    
    return (top2_size < min_size_seen_as_region)
def anatomy_filter(mask_path, select_dir):
    name = os.path.basename(mask_path)
    rough_mask_sitk = sitk.ReadImage(mask_path)
    rough_mask = sitk.GetArrayFromImage(rough_mask_sitk)
    rough_mask = postprocess(rough_mask)
    organs = LABELS.keys() - set(["background"])
    for o in organs:
        label = LABELS[o]
        if label not in [LABELS["Left Adrenal Gland"],LABELS["Right Adrenal Gland"]]:
            if np.sum(rough_mask == label)<512:
                return
        else:
            if np.sum(rough_mask == label) == 0 :

                return
            zs,ys,xs = np.where(rough_mask == label)
            if np.max(ys) - np.min(ys) < 15:
                return

        if not is_region_only_one((rough_mask == LABELS[o])):
            return
    
    if not is_region_only_one((rough_mask == LABELS["Esophagus"])|(rough_mask == LABELS["Stomach"])|(rough_mask == LABELS["Duodenum"])):
        return
    select_mask_path = os.path.join(select_dir,name)
    shutil.copy(mask_path, select_mask_path)
    in_sub = str.replace(subs["inphase"],subs["image"],subs["label"])
    if in_sub in mask_path:
        out_sub = str.replace(subs["outphase"],subs["image"],subs["label"])
        shutil.copy(mask_path,str.replace(select_mask_path,in_sub,out_sub))

def postprocess(rough_label):
    liver = rough_label == LABELS["Liver"]
    iter = 8
    ex_liver = np.zeros([s + 2*iter for s in rough_label.shape])
    ex_liver[iter:-iter,iter:-iter,iter:-iter]= liver
    ex_liver = ndi.binary_dilation(ex_liver,iterations=iter)
    zs,_,_ = np.where(ex_liver==1)
    for z in np.unique(zs):
        ex_liver[z] = ndi.binary_fill_holes(ex_liver[z])
    miss_liver = ndi.binary_erosion(ex_liver,iterations=iter)[iter:-iter,iter:-iter,iter:-iter]\
        *(rough_label!=LABELS["Gall Bladder"])*(rough_label!=LABELS["Inferior Vena Cava"])\
            *(rough_label!=LABELS["Inferior Vena Cava"])*(rough_label!=LABELS["Right Adrenal Gland"])
    rough_label[miss_liver==1]= LABELS["Liver"]



    spleen = rough_label == LABELS["Spleen"]
    iter = 8
    ex_spleen = np.zeros([s + 2*iter for s in rough_label.shape])
    ex_spleen[iter:-iter,iter:-iter,iter:-iter]= spleen
    ex_spleen = ndi.binary_dilation(ex_spleen,iterations=iter)
    zs,_,_ = np.where(ex_spleen==1)
    for z in np.unique(zs):
        ex_spleen[z] = ndi.binary_fill_holes(ex_spleen[z])
    miss_spleen = ndi.binary_erosion(ex_spleen,iterations=iter)[iter:-iter,iter:-iter,iter:-iter]\
        *(rough_label!=LABELS["Left Adrenal Gland"])*(rough_label!=LABELS["Left Kidney"])
    rough_label[miss_spleen==1]= LABELS["Spleen"]



    stomach = rough_label == LABELS["Stomach"]
    iter = 8
    ex_stomach = np.zeros([s + 2*iter for s in rough_label.shape])
    ex_stomach[iter:-iter,iter:-iter,iter:-iter]= stomach
    ex_stomach = ndi.binary_dilation(ex_stomach,iterations=iter)
    zs,_,_ = np.where(ex_stomach==1)
    for z in np.unique(zs):
        ex_stomach[z] = ndi.binary_fill_holes(ex_stomach[z])
    miss_stomach = ndi.binary_erosion(ex_stomach,iterations=iter)[iter:-iter,iter:-iter,iter:-iter]*(rough_label==0)   
    rough_label[miss_stomach==1]= LABELS["Stomach"]



    vena = rough_label == LABELS["Inferior Vena Cava"]
    structure = np.ones([8,1,1])
    iter = 8
    ex_vena = np.zeros([s + 2*iter for s in rough_label.shape])
    ex_vena[iter:-iter,iter:-iter,iter:-iter]= vena
    ex_vena = ndi.binary_dilation(ex_vena,structure=structure,iterations=1)
    miss_vena = ndi.binary_erosion(ex_vena,structure=structure,iterations=1)[iter:-iter,iter:-iter,iter:-iter]*(rough_label!=LABELS["Right Adrenal Gland"])
    rough_label[miss_vena==1]= LABELS["Inferior Vena Cava"]

    return rough_label




if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--stage2_pred', type=str, default='./data/stage2_pred')
    parser.add_argument('--stage3_label_dir', type=str, default='./data/stage3/labels')    
    parser.add_argument('--num_process', type=int, default=8)
    # 继续添加其他参数...
    args = parser.parse_args()


    if not os.path.exists(args.stage3_label_dir):
        os.makedirs(args.stage3_label_dir)
    
    mask_paths = [os.path.join(args.stage2_pred, f) for f in os.listdir(args.stage2_pred) if f.endswith(".nii.gz") ]
    ptqdm(function = anatomy_filter, iterable = mask_paths, processes = args.num_process, 
          select_dir=args.stage3_label_dir)
        
