import os 
import SimpleITK as sitk
# import torchio as tio
# import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm

input_dir = "data/origin_labels"
output_dir = "data/labels"
in_sub = ".nii.gz"
out_sub = ".nii.gz"
is_label = True
os.makedirs(output_dir, exist_ok=True)

def standardize(image_path:str, output_dir:str, is_label = False)->None:

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
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    img = resample.Execute(img)
    img.SetDirection((1,0,0,0,1,0,0,0,1))
    img.SetOrigin([0.0, 0.0, 0.0])
    img.SetSpacing(new_spacing)
    sitk.WriteImage(img, os.path.join(output_dir, name.replace(in_sub, out_sub)))




if __name__ == "__main__":
    img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(in_sub) ]
    ptqdm(function = standardize, iterable = img_paths, processes = 8, 
          output_dir=output_dir,is_label = is_label
        )
