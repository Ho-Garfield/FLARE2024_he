import os 
import SimpleITK as sitk
# import torchio as tio
# import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm



def standardize(image_path:str, output_dir:str, in_sub, out_sub, is_label)->None:
    name = os.path.basename(image_path)
    img = sitk.ReadImage(image_path)
    resample = sitk.ResampleImageFilter()
    img = sitk.DICOMOrient(img, "LPS")    
    new_spacing = (1.2,1.2,3)#x,y,z
    new_size = [int(round(old_size*sp / newsp)) for old_size, sp, newsp in zip(img.GetSize(), img.GetSpacing(), new_spacing)]
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetTransform(sitk.Transform())
    if is_label.lower()=="true":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    img = resample.Execute(img)
    img.SetDirection((1,0,0,0,1,0,0,0,1))
    img.SetOrigin([0.0, 0.0, 0.0])
    img.SetSpacing(new_spacing)
    sitk.WriteImage(img, os.path.join(output_dir, name.replace(in_sub, out_sub)))



if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input_dir', type=str, default='./FLARE24/CT/images')
    parser.add_argument('--output_dir', type=str, default='./STD/CT/images')    
    parser.add_argument('--in_sub', type=str, default=".nii.gz")
    parser.add_argument('--out_sub', type=str, default=".nii.gz")
    parser.add_argument('--is_label', type=str, default="False")
    parser.add_argument('--num_process', type=int, default=8)
    # 继续添加其他参数...
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    img_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.nii.gz') ]
    ptqdm(function = standardize, iterable = img_paths, processes = args.num_process, 
          output_dir=args.output_dir, in_sub = args.in_sub, out_sub =args.in_sub ,is_label = args.is_label
        )