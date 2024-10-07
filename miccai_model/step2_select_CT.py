import os
import SimpleITK as sitk
from acvl_utils.miscellaneous.ptqdm import ptqdm
import shutil
LABELS= {
        "background": 0,
        "Liver": 1,
        "Right Kidney": 2,
        "Right Spleen": 3,
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
subs ={
       "label":".nii.gz",
       "image":"_0000.nii.gz",
       }
def select_body(path, out_dir, all_min_z=150):
    mri = sitk.ReadImage(path)
    x,y,z = mri.GetSize()
    if z >all_min_z:
        out_path = str.replace(path,os.path.dirname(path),out_dir)
        print(out_path)
        shutil.copy(path,out_path)


def copy_mask(ct_path,mask_dir,out_dir):
    name = str.replace(os.path.basename(ct_path),subs["image"],subs['label'])
    mask_path = os.path.join(mask_dir,name)
    out_path = os.path.join(out_dir,name)
    shutil.copy(mask_path,out_path)
    


ct_dir = "/DATA_16T/MICCAI/CT2MRI/CT255/before/images_255"
mask_dir = "/DATA_16T/MICCAI/CT2MRI/CT255/before/labels"
select_dir ="/DATA_16T/MICCAI/CT2MRI/CT255/before/body"

if __name__ == "__main__":

    if not os.path.exists(os.path.join(select_dir,"masks")):
        os.makedirs(os.path.join(select_dir,"masks"))
    if not os.path.exists(os.path.join(select_dir,"images")):
        os.makedirs(os.path.join(select_dir,"images"))

    image_paths = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith(subs["image"])]

    select_paths = [os.path.join(os.path.join(select_dir,"images"), f) for f in os.listdir(os.path.join(select_dir,"images")) if f.endswith(".nii.gz")]
    ptqdm(function = copy_mask, iterable = select_paths, processes = 8,
          mask_dir=mask_dir,out_dir=os.path.join(select_dir,"masks")
    ) 
