import os
import glob

def rename_nii_files(directory):
    # Get all .nii.gz files in the directory
    nii_files = glob.glob(os.path.join(directory, '*.nii.gz'))

    # Sort the files to ensure consistent ordering
    nii_files.sort()

    # Loop through the files and rename them
    for i, file in enumerate(nii_files, start=1):
        new_filename = f"image{i}.nii.gz"
        new_filepath = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(file, new_filepath)
        print(f"Renamed {file} to {new_filepath}")

# Specify the directory containing the .nii.gz files
directory = '/DATA_16T/MICCAI/Data_GAN/source'
rename_nii_files(directory)