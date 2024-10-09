
import ants
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy as np
from skimage import filters
tmp_dir = r"./temp1"

img_sub = "_0000.nii.gz"
lab_sub = ".nii.gz"

a0_path ="./images_255_select/FLARE22_Tr_0001_0000.nii.gz"
a0_label_path ="./labels/FLARE22_Tr_0001.nii.gz"
b0_path = "./Flip_resample/MR745_6_C-pre_0000.nii.gz"

out_dir_mid = ""
out_dir_final = ""

input_images_dir="./images_255_select"
input_labels_dir ="./labels"
input_targets_dir="./Flip_resample"



if(not os.path.exists(tmp_dir)):
    os.makedirs(tmp_dir,exist_ok=True)
def clean_tmp(contain_str:str, tmp_dir = r"./temp")->None:
    for name in os.listdir(tmp_dir) :
        if contain_str in name:
            os.remove(os.path.join(tmp_dir, name) )

def registration_a2b(a_path,b_path,i, label_dir, out_dir):
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
    a_label_directory = os.path.join(str(out_dir), 'labels',str(i) + '.nii')
    b__directory = os.path.join(str(out_dir), 'targets', str(i) + '.nii')
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



if __name__ == "__main__":
    if not os.path.exists(os.path.join(out_dir_mid,'images')):
        os.makedirs(os.path.join(out_dir_mid,'images'))
        os.makedirs(os.path.join(out_dir_mid,'labels'))
        os.makedirs(os.path.join(out_dir_mid,'targets'))


    registration_a02b0(a0_path,b0_path,a0_label_path,out_dir_mid)
    new_a0_path = os.path.join(out_dir_mid,'images',os.path.basename(a0_path))
    new_a0_label_path = os.path.join(out_dir_mid,'labels',os.path.basename(a0_label_path))

    img_paths = [os.path.join(input_targets_dir, f) for f in os.listdir(input_targets_dir) if f.endswith("_C-pre_0000.nii.gz") and f not in os.listdir(os.path.join(out_dir_mid,'targets'))]
    num = range(0,len(img_paths))
    ptqdm(function = registration_b_b0, iterable = img_paths, processes = 6,
        refer_path=b0_path,out_dir=out_dir_mid
    )
    img_paths = [os.path.join(input_images_dir, f) for f in os.listdir(input_images_dir) if f.endswith(img_sub) and f not in os.listdir(os.path.join(out_dir_mid,'images'))]
    num = range(0,len(img_paths))
    ptqdm(function = registration_a_witha0_label, iterable = img_paths, processes = 3,
          label_dir=input_labels_dir,refer_path=new_a0_label_path,out_dir=out_dir_mid
        )
    if not os.path.exists(os.path.join(out_dir_final,'images')):
        os.makedirs(os.path.join(out_dir_final,'images'))
        os.makedirs(os.path.join(out_dir_final,'labels'))
        os.makedirs(os.path.join(out_dir_final,'targets'))

    new_a_dir = os.path.join(out_dir_mid,"images")
    new_a_label_dir = os.path.join(out_dir_mid,"labels")
    new_b_dir = os.path.join(out_dir_mid,"targets")
    a_paths = [os.path.join(new_a_dir, f) for f in os.listdir(new_a_dir) if f.endswith(img_sub)]
    b_paths = [os.path.join(new_b_dir, f) for f in os.listdir(new_b_dir) if f.endswith(img_sub)]
    i = range(0,len(a_paths))
    ptqdm(function = registration_a2b, iterable = (a_paths,b_paths,i), processes = 3,
          zipped=True,label_dir=new_a_label_dir,out_dir=out_dir_final
        )
