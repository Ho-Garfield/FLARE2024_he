
import SimpleITK as sitk
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm

input_dir = "/media/wl/Extreme SSD/3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train/fake_mri"
out_dir ="/media/wl/Extreme SSD/3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train/fake_mri_gz"
def CTNormalization(image_path,min,max,out_dir):
    image = sitk.ReadImage(image_path)
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(max)
    intensityWindowingFilter.SetWindowMinimum(min)
    image = intensityWindowingFilter.Execute(image)
    sitk.WriteImage(image,os.path.join(out_dir,os.path.basename(image_path)))

def MRINormalization(image_path,out_dir):
    image = sitk.ReadImage(image_path)
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)

    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-255
    sitk.WriteImage(image,os.path.join(out_dir,os.path.basename(image_path)))

def nii_nii_gz_switch(f_path,out_dir):
    s_file = sitk.ReadImage(f_path)
    if ".gz" not in f_path:
        s = ".nii"
        t =".nii.gz"
    else:
        s = ".nii.gz"
        t =".nii"

    name = str.replace(os.path.basename(f_path),s,t)
    out_path = os.path.join(out_dir,name)
    sitk.WriteImage(s_file,out_path)




if __name__ == "__main__":
    os.makedirs(out_dir,exist_ok=True)
    img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nii") ]
    # ptqdm(function = nii_nii_gz_switch, iterable = img_paths, processes = 8,out_dir=out_dir)
    ptqdm(function = CTNormalization, iterable = img_paths, processes = 4, min = -160, max=240,out_dir=out_dir)
