import tkinter as tk
from tkinter import messagebox
import subprocess
import os

# Set the script directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Define the parameters for each script
PARAMS = {
    "./standardized.py": {
        "input_dir": {"default": './FLARE24/CT/images', "type": str},
        "output_dir": {"default": './STD/CT/images', "type": str},
        "in_sub": {"default": '.nii.gz', "type": str},
        "out_sub": {"default": '.nii.gz', "type": str},
        "is_label": {"default": 'False', "type": str},
        "num_process": {"default": 8, "type": int}
    },
    "./flip_mri_with_inphase.py": {
        "inphase_dir": {"default": './STD/MRI/LLD', "type": str},
        "handle_dir": {"default": './STD/MRI/LLD', "type": str},
        "out_dir": {"default": './STD/MRI/LLD', "type": str},
        "num_process": {"default": 4, "type": int}
    },
    "./gray2255.py": {
        "input_dir": {"default": './STD/CT/images', "type": str},
        "out_dir": {"default": './STD/CT/images_255', "type": str},
        "is_CT": {"default": 'True', "type": str},
        "num_process": {"default": 8, "type": int}
    },
    "./registration_ct_mri.py": {
        "T1W_dir": {"default": './STD/MRI/LLD_255', "type": str},
        "CT_images_dir": {"default": './STD/CT/images_255', "type": str},
        "CT_labels_dir": {"default": './STD/CT/labels', "type": str},
        "out_dir_mid_temp": {"default": './STD/temp', "type": str},
        "out_dir": {"default": './3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train', "type": str},  
        "stage1_data_dir": {"default": './data/stage1', "type": str},  

              
        "img_sub": {"default": '_0000.nii.gz', "type": str},
        "lab_sub": {"default": '.nii.gz', "type": str},
        "num_process": {"default": 4, "type": int}
    },
    "3D-CycleGan-Pytorch-MedImaging-main/gan_train.py": {
        "data_path": {"default": './3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train', "type": str},
    },
    "3D-CycleGan-Pytorch-MedImaging-main/gan_pred.py": {
        "input_dir": {"default": './3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train/images', "type": str},
        "out_dir": {"default": './data/stage1/images', "type": str}
    },
    "semi/train.py": {
        "root_path": {"default": './data/stage1/', "type": str},
        "model":{"default": 'semi_model', "type": str},
        
    },
    "semi/predict.py": {
        "model_path": {"default": './model/Mean_Teacher_f1/semi_model/best_model.pth', "type": str},
        "image_folder": {"default": './STD/MRI/LLD', "type": str},
        "predict_folder": {"default": './data/stage1_pred_t1w', "type": str},
        "in_sub": {"default": '_C-pre_0000.nii.gz', "type": str}
    },
    "./registration_self.py": {
        "in_dir": {"default": './STD/MRI/LLD', "type": str},
        "out_dir": {"default": './data/stage2/images', "type": str},
        "num_process": {"default": 5, "type": int}
    },

    "LLD_label_share_kidney_match.py": {
        "mris_dir": {"default": './data/stage2/images', "type": str},
        "mask_dir": {"default": './data/stage1_pred_t1w', "type": str},
        "temp_dir": {"default": './data/stage2_temp', "type": str},
        "final_label_dir": {"default": 'data/stage2/labels', "type": str},
        "num_process": {"default": 8, "type": int}
    },
    "select_CT.py": {
        "ct_255_dir": {"default": './STD/CT/images_255', "type": str},
        "ct_labels_dir": {"default": './STD/CT/labels', "type": str},
        "stage2_data_dir": {"default": './data/stage2/', "type": str},
        "all_min_z": {"default": 150, "type": int},
        "contain_letter": {"default": 'T', "type": str},
        "description": {"default": 'select CT sample with all_min_z or contain_letter', "type": str},
        "num_process": {"default": 8, "type": int}
    },
    "anatomy_filter.py": {
        "stage2_pred": {"default": './data/stage2_pred', "type": str},
        "stage3_label_dir": {"default": './data/stage3/labels', "type": str},
        "num_process": {"default": 8, "type": int}
    },



}

# Map stages to scripts
STAGES = {
    "Preprocess":["./standardized.py", "./flip_mri_with_inphase.py","./gray2255.py","./registration_ct_mri.py"],
    "Stage 1": ["3D-CycleGan-Pytorch-MedImaging-main/gan_train.py", "3D-CycleGan-Pytorch-MedImaging-main/gan_pred.py",
                "semi/train.py","semi/predict.py"],
    "Stage 2": ["./registration_self.py", "LLD_label_share_kidney_match.py","select_CT.py",
                "semi/train.py","semi/predict.py"],
    "Stage 3": ["anatomy_filter.py", "select_CT.py","semi/train.py","semi/predict.py"],
}

# Function to run the selected script
def run_script(script, params):
    try:
        process = subprocess.run(["python", script] + params)
        messagebox.showinfo("Success", f"{script} executed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", f"Failed to execute {script}.")

# Function to switch frames based on selected script
def switch_frame(stage_name, script_name):
    for widget in frame.winfo_children():
        widget.destroy()

    entries = {}
    script_params = PARAMS.get(script_name, {})
    tk.Label(frame, text=f"*******************{stage_name.capitalize()}*******************").pack(pady=5)
    tk.Label(frame, text=f"****{(script_name.split('/')[-1]).capitalize()}****").pack(pady=5)

    for param, options in script_params.items():
        tk.Label(frame, text=f"{param.replace('_', ' ').capitalize()}:").pack(pady=5)
        entry = tk.Entry(frame, width=50)
        entry.insert(0, str(options["default"]))
        entry.pack(pady=5)
        entries[param] = entry

    run_button = tk.Button(frame, text="Run", command=lambda: run_script(script_name, [
        f"--{param}={entries[param].get()}" for param in script_params
    ]))
    run_button.pack(pady=10)

# Function to show scripts for the selected stage
def show_scripts(stage_name):
    for widget in frame.winfo_children():
        widget.destroy()

    tk.Label(frame, text=f"*************{stage_name} Scripts*************").pack(pady=5)

    # Create buttons for each script in the selected stage
    for script_name in STAGES[stage_name]:
        name = script_name.split('/')[-1]
        button = tk.Button(frame, text=name, command=lambda sn=script_name: switch_frame(stage_name, sn))
        button.pack(pady=5)

# Main window setup
root = tk.Tk()
root.title("Script Runner")

# Top button frame for stages
stage_frame = tk.Frame(root)
stage_frame.pack(side=tk.TOP, fill=tk.X)

# Define stage buttons
for stage_name in STAGES.keys():
    button = tk.Button(stage_frame, text=stage_name, command=lambda sn=stage_name: show_scripts(sn))
    button.pack(side=tk.LEFT, padx=5, pady=5)

# Main content area
frame = tk.Frame(root)
frame.pack(pady=20)

# Start the main loop
root.mainloop()
