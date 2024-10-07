import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import itertools
import logging
import scipy.ndimage as ndi
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, select_label_files=None, transform=None, 
                 img_suffix="_0000.nii.gz", mask_suffix=".nii.gz", is_semi = True, use_half_label=False,\
                    unlabel_transform=None, unlabel_files=None, all_label_files=None):
        """
        - Parameters:
            - root_dir:root directory contian folders:'images','labels','half_labels'
            - select_label_files: labeled sample file name selected for supervised training
            - use_half_label: use label in folder 'half_labels' for cropping
            - unlabel_transform: If it's not None, use different enhancements for labeled and unlabeled respectively
            - unlabel_files: File names without labels in the 'images' directory. 
                If it is None, it is a collection of file names that exist in 'images' but do not have corresponding labels in the 'labels' directory

        """
        self.img_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.half_label_dir = os.path.join(root_dir, 'half_labels')    
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix


        self.use_half_label = use_half_label
        self.label_transform = transform
        self.unlabel_transform = unlabel_transform
        
        if all_label_files is None:
            all_label_files = [f.replace(mask_suffix, img_suffix) for f in os.listdir(self.label_dir) if f.endswith(mask_suffix)]
        if not is_semi:
            unlabel_files = []
        elif is_semi and unlabel_files is None:# semi supervised
            # 获取数据文件列表
            unlabel_files = [f for f in os.listdir(self.img_dir) if f.endswith(img_suffix) and f not in all_label_files]
        logging.info(f"ALL : {len(all_label_files + unlabel_files)}")

        if select_label_files is None:#use all labeled files for training
            self.label_num = len(all_label_files)
            self.sample_files = all_label_files + unlabel_files
            logging.info(f"SELECT : {len(self.sample_files)} "+
            f"SELECT LABEL: {self.label_num}")
        else:
            self.label_num = len(select_label_files)
            self.sample_files = select_label_files + unlabel_files
            logging.info(f"SELECT : {len(self.sample_files)}(of {len(all_label_files + unlabel_files)}), "+
                        f"SELECT LABEL: {len(select_label_files)}(of {len(all_label_files)})")


    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):

        sample_file = self.sample_files[idx]
        # 读取数据和标签
        image_sitk = sitk.ReadImage(os.path.join(self.img_dir, sample_file))
        #(z, y, x)
        image =sitk.GetArrayFromImage(image_sitk)
        label = np.zeros(image.shape)
        if(idx < self.label_num):
            label_path = os.path.join(self.label_dir, sample_file.replace(self.img_suffix, self.mask_suffix))
            label_sitk = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label_sitk)
        elif(self.use_half_label):
            label_path = os.path.join(self.half_label_dir, sample_file.replace(self.img_suffix, self.mask_suffix))
            label_sitk = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label_sitk)
        label = label*(image>0)
        is_diff_trans = self.unlabel_transform is not None
        if is_diff_trans and not (idx < self.label_num):
            sample = {'image': image, 'label': label}
            sample = self.unlabel_transform(sample)

        else:
            sample = {'image': image, 'label': label}
            if (self.label_transform is not None):
                sample = self.label_transform(sample)

        return sample
            

            
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        """
        for data in dataloader: 时调用,返回对应索引列表 调用dataset的__getitem__(index)
        """
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch# 并集
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    """
    - Return (ndarray): 对传进来可迭代对象进行一次随机排
    """
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    """
    - Return (ndarray): 对传进来迭代对象执行以下操作,每对于该对象迭代一次时,都对迭代对象随机重排
    """
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)