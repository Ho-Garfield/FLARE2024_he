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
    



if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--ct_255_dir', type=str, default='./STD/CT/images_255')
    parser.add_argument('--ct_labels_dir', type=str, default='./STD/CT/labels')
    parser.add_argument('--stage2_data_dir', type=str, default='./data/stage2/')  
    parser.add_argument('--all_min_z', type=int, default=150) 
    parser.add_argument('--contain_letter', type=str, default='T') 
    parser.add_argument('--description', type=str, default='select CT sample with all_min_z or contain_letter') 
    parser.add_argument('--num_process', type=int, default=8)
    # 继续添加其他参数...
    args = parser.parse_args()


    if not os.path.exists(os.path.join(args.stage2_data_dir,"labels")):
        os.makedirs(os.path.join(args.stage2_data_dir,"labels"))
    if not os.path.exists(os.path.join(args.stage2_data_dir,"images")):
        os.makedirs(os.path.join(args.stage2_data_dir,"images"))

    image_paths = [os.path.join(args.ct_255_dir, f) for f in os.listdir(args.ct_255_dir) if f.endswith(subs["image"])]
    ptqdm(function = select_body, iterable = image_paths, processes = args.num_process,
          out_dir=os.path.join(args.stage2_data_dir,"images"),all_min_z=args.all_min_z
    ) 
    list_ct = os.listdir(args.ct_255_dir)
    select_paths = [os.path.join(os.path.join(args.stage2_data_dir,"images"), f) for f in os.listdir(os.path.join(args.stage2_data_dir,"images")) if f.endswith(".nii.gz") and f in list_ct]
    select_paths2 = [os.path.join(args.ct_255_dir, f) for f in os.listdir(args.ct_255_dir) if f.endswith(subs["image"]) and args.contain_letter in f]
    ptqdm(function = select_body, iterable = select_paths2, processes = args.num_process,
          out_dir=os.path.join(args.stage2_data_dir,"images"),all_min_z=0
    ) 
    select_paths = list(set(select_paths).union(set(select_paths2)))
    ptqdm(function = copy_mask, iterable = select_paths, processes = args.num_process,
          mask_dir=args.ct_labels_dir,out_dir=os.path.join(args.stage2_data_dir,"labels")
    ) 




