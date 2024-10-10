
import ants
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy as np
from skimage import filters
tmp_dir = r"./temp1"

img_sub = ""
lab_sub = ""




if(not os.path.exists(tmp_dir)):
    os.makedirs(tmp_dir,exist_ok=True)
def clean_tmp(contain_str:str, tmp_dir = r"./temp")->None:
    for name in os.listdir(tmp_dir) :
        if contain_str in name:
            os.remove(os.path.join(tmp_dir, name) )

def registration_a2b(a_path,b_path,i, label_dir, out_dir,stage1_data_dir):
    b = ants.image_read(b_path)
    a = ants.image_read(a_path)
    a_name = os.path.basename(a_path)    
    lab_name = str.replace(a_name,img_sub,lab_sub)
    a_label = ants.image_read(os.path.join(label_dir,lab_name))
    
    anp=a.numpy()
    bnp=b.numpy()

    t = filters.threshold_otsu(anp)
    mask_a = anp>t
    mask_a_sitk = ants.from_numpy(mask_a.astype(np.float32),a.origin,a.spacing,a.direction)
    t = filters.threshold_otsu(bnp)
    mask_b = bnp>t
    mask_b_sitk = ants.from_numpy(mask_b.astype(np.float32),b.origin,b.spacing,b.direction)


    ra2b = ants.registration(mask_b_sitk , mask_a_sitk, type_of_transform="Translation",random_seed=42,
                            outprefix = os.path.join(tmp_dir, a_name))
    a =ants.apply_transforms(fixed=b, moving=a, transformlist=ra2b['fwdtransforms'])
    a_label =ants.apply_transforms(fixed=b, moving=a_label, transformlist=ra2b['fwdtransforms'],interpolator="nearestNeighbor")
    
    clean_tmp(a_name, tmp_dir=tmp_dir)
    
    
    a_directory = os.path.join(str(out_dir), 'images', str(i) + '.nii')
    a_label_directory = os.path.join(str(stage1_data_dir), 'labels',str(i) + '.nii.gz')
    b__directory = os.path.join(str(out_dir), 'labels', str(i) + '.nii')
    ants.image_write(a, a_directory)
    ants.image_write(a_label, a_label_directory)
    ants.image_write(b, b__directory)

def registration_a_witha0_label(img_path,label_dir,refer_path,out_dir):
    img_name = os.path.basename(img_path)
    lab_name = str.replace(img_name,img_sub,lab_sub)
    image = ants.image_read(img_path)  
    label = ants.image_read(os.path.join(label_dir,lab_name))

    refer0 = ants.image_read(refer_path)

    r20 = ants.registration(refer0, label, type_of_transform="Translation",random_seed=42,
                            outprefix = os.path.join(tmp_dir, lab_name))
    label =ants.apply_transforms(fixed=refer0, moving=label, transformlist=r20['fwdtransforms'],interpolator='nearestNeighbor')
    image =ants.apply_transforms(fixed=refer0, moving=image, transformlist=r20['fwdtransforms'])
    
    clean_tmp(lab_name, tmp_dir=tmp_dir)
    label_directory = os.path.join(str(out_dir), 'labels',lab_name)
    image_directory = os.path.join(str(out_dir), 'images',img_name)

    ants.image_write(image, image_directory)
    ants.image_write(label, label_directory)

def registration_b_b0(img_path,refer_path,out_dir):
    img_name = os.path.basename(img_path)
    image = ants.image_read(img_path)  

    refer0 = ants.image_read(refer_path)


    inp=image.numpy()
    rnp=refer0.numpy()
    p_80 = np.percentile(inp,80)
    t = filters.threshold_otsu(inp[inp<p_80])
    mask_i = inp>t
    mask_i_sitk = ants.from_numpy(mask_i.astype(np.float32),image.origin,image.spacing,image.direction)
    p_80 = np.percentile(rnp,80)
    t = filters.threshold_otsu(rnp[rnp<p_80])

    mask_r = rnp>t
    mask_r_sitk = ants.from_numpy(mask_r.astype(np.float32),refer0.origin,refer0.spacing,refer0.direction)

    r20 = ants.registration(mask_r_sitk , mask_i_sitk, type_of_transform="Translation",random_seed=42,
                            outprefix = os.path.join(tmp_dir, img_name))

    image =ants.apply_transforms(fixed=mask_r_sitk, moving=image, transformlist=r20['fwdtransforms'])
    
    clean_tmp(img_name, tmp_dir=tmp_dir)

        
    image_directory = os.path.join(str(out_dir), 'targets',img_name)

    ants.image_write(image, image_directory)

def registration_a02b0(a0_path,b0_path,a0_label_path,a_out_dir):
    a_name = os.path.basename(a0_path)
    a = ants.image_read(a0_path)  
    a_label = ants.image_read(a0_label_path)    
    b = ants.image_read(b0_path)

    anp=a.numpy()
    bnp=b.numpy()

    t = filters.threshold_otsu(anp)
    mask_a = anp>t
    mask_a_sitk = ants.from_numpy(mask_a.astype(np.float32),a.origin,a.spacing,a.direction)
    t = filters.threshold_otsu(bnp)
    mask_b = bnp>t
    mask_b_sitk = ants.from_numpy(mask_b.astype(np.float32),b.origin,b.spacing,b.direction)

    ra2b = ants.registration(mask_b_sitk , mask_a_sitk, type_of_transform="Translation",random_seed=42,
                            outprefix = os.path.join(tmp_dir, a_name))
    label =ants.apply_transforms(fixed=mask_b_sitk, moving=a_label, transformlist=ra2b['fwdtransforms'],interpolator='nearestNeighbor')
    mvimage =ants.apply_transforms(fixed=mask_b_sitk, moving=a, transformlist=ra2b['fwdtransforms'])
    clean_tmp(a_name, tmp_dir=tmp_dir)

        
    label_directory = os.path.join(str(a_out_dir), 'labels', os.path.basename(a0_label_path))
    image_directory = os.path.join(str(a_out_dir), 'images', a_name)

    ants.image_write(mvimage, image_directory)
    ants.image_write(label, label_directory)



import ants




if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--T1W_dir', type=str, default='./STD/MRI/LLD_255')
    parser.add_argument('--CT_images_dir', type=str, default='./STD/CT/images_255')
    parser.add_argument('--CT_labels_dir', type=str, default='./STD/CT/labels')
    parser.add_argument('--out_dir_mid_temp', type=str, default='./STD/temp')
    parser.add_argument('--out_dir', type=str, default='./3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train')
    parser.add_argument('--stage1_data_dir', type=str, default='./data/stage1')

    parser.add_argument('--img_sub', type=str, default="_0000.nii.gz")
    parser.add_argument('--lab_sub', type=str, default=".nii.gz")
    parser.add_argument('--num_process', type=int, default=4)
    # 继续添加其他参数...
    args = parser.parse_args()
    img_sub = args.img_sub
    lab_sub = args.lab_sub
    if not os.path.exists(os.path.join(args.out_dir_mid_temp,'images')):
        os.makedirs(os.path.join(args.out_dir_mid_temp,'images'))
        os.makedirs(os.path.join(args.out_dir_mid_temp,'labels'))
        os.makedirs(os.path.join(args.out_dir_mid_temp,'targets'))
        

    a0_name = os.listdir(args.CT_images_dir)[0]
    a0_path = os.path.join(args.CT_images_dir,a0_name)
    a0_label_path = os.path.join(args.CT_labels_dir,str.replace(a0_name,img_sub,lab_sub))
    b0_path = os.path.join(args.T1W_dir,os.listdir(args.T1W_dir)[0])
    print(f"using a0: {a0_path}, b0: {b0_path}")
    #a0-->b0
    registration_a02b0(a0_path,b0_path,a0_label_path,args.out_dir_mid_temp)
    new_a0_path = os.path.join(args.out_dir_mid_temp,'images',os.path.basename(a0_path))
    new_a0_label_path = os.path.join(args.out_dir_mid_temp,'labels',os.path.basename(a0_label_path))

    img_paths = [os.path.join(args.T1W_dir, f) for f in os.listdir(args.T1W_dir) if f.endswith("_C-pre_0000.nii.gz") and f not in os.listdir(os.path.join(args.out_dir_mid_temp,'targets'))]
    os.makedirs(os.path.join(args.stage1_data_dir,'images'),exist_ok=True)
    import shutil
    for img_path in img_paths:
        out_path = os.path.join(args.stage1_data_dir,'images',os.path.basename(img_path))
        if not os.path.exists(out_path):
            shutil.copy2(img_path,out_path)
    #b-->b0
    num = range(0,len(img_paths))
    ptqdm(function = registration_b_b0, iterable = img_paths, processes = args.num_process,
        refer_path=b0_path,out_dir=args.out_dir_mid_temp
    )
    #a-->a0
    img_paths = [os.path.join(args.CT_images_dir, f) for f in os.listdir(args.CT_images_dir) if f.endswith(img_sub) and f not in os.listdir(os.path.join(args.out_dir_mid_temp,'images'))]
    num = range(0,len(img_paths))
    ptqdm(function = registration_a_witha0_label, iterable = img_paths, processes = args.num_process,
          label_dir=args.CT_labels_dir,refer_path=new_a0_label_path,out_dir=args.out_dir_mid_temp
        )
    os.makedirs(os.path.join(args.out_dir,'images'),exist_ok=True)
    os.makedirs(os.path.join(args.stage1_data_dir,'labels'),exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'labels'),exist_ok=True)

    new_a_dir = os.path.join(args.out_dir_mid_temp,"images")
    new_a_label_dir = os.path.join(args.out_dir_mid_temp,"labels")
    new_b_dir = os.path.join(args.out_dir_mid_temp,"targets")
    a_paths = [os.path.join(new_a_dir, f) for f in os.listdir(new_a_dir) if f.endswith(img_sub)]
    b_paths = [os.path.join(new_b_dir, f) for f in os.listdir(new_b_dir) if f.endswith(img_sub)]
    i = range(0,len(a_paths))
    #a-->b
    ptqdm(function = registration_a2b, iterable = (a_paths,b_paths,i), processes = args.num_process,
          zipped=True,label_dir=new_a_label_dir,out_dir=args.out_dir,stage1_data_dir=args.stage1_data_dir
        )
    shutil.rmtree(args.out_dir_mid_temp)
