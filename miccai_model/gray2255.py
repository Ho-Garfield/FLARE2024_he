
import SimpleITK as sitk
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm


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





if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input_dir', type=str, default='./STD/CT/images')
    parser.add_argument('--out_dir', type=str, default='./STD/CT/images_255')    
    parser.add_argument('--is_CT', type=str, default="True")
    parser.add_argument('--num_process', type=int, default=8)
    # 继续添加其他参数...
    args = parser.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    img_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".nii.gz") ]
    if args.is_CT.lower()=="true":
        ptqdm(function = CTNormalization, iterable = img_paths, processes = args.num_process, min = -160, max=240,out_dir=args.out_dir)
    else:
        ptqdm(function = MRINormalization, iterable = img_paths, processes = args.num_process, out_dir=args.out_dir)

