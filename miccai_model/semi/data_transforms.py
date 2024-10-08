import random
import numpy as np
from typing import List, Callable,Tuple
import numpy as np
import config
import torch
import scipy.ndimage as ndi
import monai.transforms as mt
class SampleRandomScale:
    def __init__(self, scale_range=(0.7,1.4), scale_percentage = 0.2) -> None:
        self.scale_range = scale_range
        self.scale_percentage = scale_percentage
    def __call__(self, sample):
        if(random.random()<self.scale_percentage):
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            zoom = mt.Zoom(zoom=scale_factor,keep_size=False)
            for key in sample.keys():
                image_depth, image_height, image_width= sample[key].shape
                if key == 'label':
                    #zoom image shape:(C,Z,X,Y)                
                    sample[key] = zoom(sample[key].reshape(1,*sample[key].shape), mode="nearest").squeeze(0).numpy()
                else:
                    sample[key] = zoom(sample[key].reshape(1,*sample[key].shape), mode="trilinear").squeeze(0).numpy()

            #print("do scale")
            return sample
        else:
            return sample

class CurrentLabelBatch():
    def __init__(self, complete_label_batch_size) -> None:
        self.cur_batch = 0
        self.label_batch_size = complete_label_batch_size
        
    def get_cur_bath(self):
        return self.cur_batch

    def iter(self):
        self.cur_batch = (self.cur_batch + 1)%self.label_batch_size
    def get_batch_size(self):
        return self.label_batch_size

class SampleRandomCrop:
    def __init__(self, output_size, remove_z_side = False,foreground_batch_percentage = 0.33, 
                 current_label_batch:Tuple[CurrentLabelBatch,None] = None,
                 current_unlabel_batch:Tuple[CurrentLabelBatch,None] = None,
                 incomplete_class_num = 1,
                 foreground_labels=None,
                 ) -> None:
        """
        - Parameters:
            - foreground_batch_percentage: The proportion of samples containing foreground labels after cropping in a batch
            - incomplete_class_num: num of incomplete label, if unlabel(all zero), incomplete_class_num = 1
            - foreground_labels: list of foreground labels
        """
        self.output_size = output_size
        self.foreground_batch_percentage = foreground_batch_percentage
        self.remove_z_side = remove_z_side
        self.remove_z_side_percent = 0.1
        self.current_label_batch = current_label_batch
        self.current_unlabel_batch = current_unlabel_batch
        self.incomplete_class_num = incomplete_class_num 
        self.foreground_labels =foreground_labels
    def __call__(self, sample):
        if not('label' in sample):
            assert 'image' in sample, 'missing image for random_crop_threshold'
        depth, height, width  = self.output_size
        for key in sample.keys():
            image_depth, image_height, image_width= sample[key].shape

            if image_width < width or image_height < height or image_depth < depth:
                pad_depth = max(depth - image_depth, 0)
                pad_height = max(height - image_height, 0)
                pad_width = max(width - image_width, 0)
                pad_value = 0  
                sample[key] = np.pad(sample[key], ((0, pad_depth), (0, pad_height), (0, pad_width)), mode='constant', constant_values=pad_value)
        if self.remove_z_side:
            return self.random_crop(sample)
        return self.random_foreground_crop(sample)


    def random_crop(self, sample):        
        depth, height, width  = self.output_size 
        image_depth, image_height, image_width= sample['image'].shape
        if(self.remove_z_side):
            # make sure the size after removing the sides is bigger than output size
            self.remove_z_side_percent = min(self.remove_z_side_percent,(image_depth - depth)/(2*image_depth))
            z_begin = int(image_depth*self.remove_z_side_percent)
            z_end = image_depth - int(image_depth*self.remove_z_side_percent) - depth
            if z_begin < z_end:
                z_start = random.randint(z_begin, z_end)
            else:
                z_start = z_begin

        else:
            z_start = random.randint(0, image_depth - depth)    
        y_start = random.randint(0, image_height - height)
        x_start = random.randint(0, image_width - width)
            
        for key in sample.keys():
            sample[key] = sample[key][z_start:z_start + depth, y_start:y_start + height, x_start:x_start + width]
            sample[key] = sample[key].reshape(self.output_size)
        return sample
    
    def random_foreground_crop(self, sample): 
        # #print("do random_foreground_crop")
        if 'label' not in sample.keys():
            return self.random_crop_threshold(sample) 

        label_for_crop = np.zeros(sample['label'].shape)    
        if self.foreground_labels is not None:
            for l,i in zip(self.foreground_labels,range(1,len(self.foreground_labels)+1)):
                label_for_crop[sample['label']==l] = i
        else:
            label_for_crop = sample['label']
        label_max = np.max(label_for_crop)
        num_class = int(label_max + 1)
        if num_class <= self.incomplete_class_num:# unlabel batch
            if self.current_unlabel_batch is None:
                return self.random_crop_threshold(sample) 
            else:   
                cur_batch = float(self.current_unlabel_batch.get_cur_bath())
                batch_size = float(self.current_unlabel_batch.get_batch_size())
                self.current_unlabel_batch.iter()
        else:      
            if self.current_label_batch is None:
                return self.random_crop_threshold(sample) 
            else:         
                cur_batch = float(self.current_label_batch.get_cur_bath())
                batch_size = float(self.current_label_batch.get_batch_size())
                self.current_label_batch.iter()
        if num_class <= 1:#label all zero
            return self.random_crop_threshold(sample) 
        if(cur_batch/ batch_size > self.foreground_batch_percentage or batch_size==1):
            return self.random_crop_threshold(sample) 


        data_list = []
        key_list = []
        index = 0
        for key in sample.keys():
            key_list.append((key,index))
            index = index + 1
            data_list.append(sample[key]) 
        data_stack= np.stack(data_list,axis=0)
        n, image_depth, image_height, image_width= data_stack.shape 
        
        np.random.rand()
        rs = np.random.RandomState()
        rs.set_state(np.random.get_state())
        
        sample_ratios = [0] + [1/(label_max)]*int(label_max)
        # print(sample_ratios)
        crop_foreground = mt.RandCropByLabelClasses(self.output_size,ratios = sample_ratios,num_classes=num_class).set_random_state(state=rs)
        result = crop_foreground(data_stack,label_for_crop.reshape(1,*label_for_crop.shape))[0]
        n, image_depth, image_height, image_width= result.shape 

        for key,index in key_list:
            sample[key] = result[index].numpy()

        # # #print(image.shape) 
        return sample
     
    def random_crop_threshold(self, sample,threshold=None):
        image = sample['image']
        if threshold is None:
            threshold = np.min(image)
        weight = ((image > threshold).astype(np.float32))
        weight = (weight/np.sum(weight)).reshape(1,*image.shape)
        np.random.rand()
        rs = np.random.RandomState()
        rs.set_state(np.random.get_state())
        weight_crop = mt.RandWeightedCrop(self.output_size,weight_map=weight).set_random_state(state=rs)
        data_list = []
        key_list = []
        index = 0
        shape = sample['label'].shape
        for key in sample.keys():

            key_list.append((key,index))
            index = index + 1
            data_list.append(sample[key]) 
            if shape != sample[key].shape:
                print(shape,"!=",sample[key].shape)

        
        data_stack= np.stack(data_list,axis=0)  
        n, image_depth, image_height, image_width= data_stack.shape 
        result = weight_crop(data_stack)[0]
        n, image_depth, image_height, image_width = result.shape

        for key,index in key_list:
            sample[key] = result[index].numpy()

        return sample         

class SampleRandomRotateZ:
    def __init__(self, max_angle = 90, percentage = 0.2) -> None:
        self.percentage = percentage
        self.angle = max_angle
    def __call__(self, sample):
        radian = np.deg2rad(self.angle)
        np.random.rand()
        rs = np.random.RandomState()
        rs.set_state(np.random.get_state())
        rotate = mt.RandRotate(range_x=radian,prob=self.percentage,keep_size=False).set_random_state(state=rs)
        start_rotate = True
        for key in sample.keys():
            if key == 'label':
                sample['label'] = rotate(sample['label'], mode="nearest", randomize= start_rotate).numpy()
                start_rotate = False
            else:
                sample[key] = rotate(sample[key], mode="bilinear",randomize=start_rotate).numpy()
                start_rotate = False

        return sample

class SampleResizeyx:
    def __init__(self, target_yx_size) -> None:
        self.target_size = target_yx_size    
    def __call__(self, sample):
        ny, nx = self.target_size

        for key in sample.keys():
            z, y, x = sample[key].shape
            zoom_factors = (1, ny/y, nx/x)
            zoom = mt.Zoom(zoom=zoom_factors,keep_size=False)  
            if key == 'label':
                if(np.max(sample['label'])== 0):
                    sample['label'] = np.zeros((z, ny, nx))
                else:
                    sample['label'] = zoom(sample['label'].reshape(1,*sample['label'].shape),mode="nearest").squeeze(0).numpy()
            else:
                sample[key] = zoom(sample[key].reshape(1,*sample[key].shape), mode="trilinear").squeeze(0).numpy()
        return sample
    
class SampleRandomFlip:
    def __init__(self, flip_axis = (0,1,2)) -> None:
        self.flip_axis = flip_axis    
    def __call__(self, sample):
        data_list = []
        key_list = []
        index = 0
        for key in sample.keys():
            key_list.append((key,index))
            index = index + 1
            data_list.append(sample[key]) 
        data_stack= np.stack(data_list,axis=0)  
        n, image_depth, image_height, image_width= data_stack.shape 
        for axis in self.flip_axis:
            if random.random() > 0.5:
                data_stack = np.flip(data_stack, axis=axis+1)
        for key,index in key_list:
            sample[key] = data_stack[index].copy()


        return sample
 
class Sample_Normalize:
    def __init__(self, method="z-score", histogram_bins=40, diff =False, teacher_key="image_ema", student_key="image") -> None:
        """
        - Parameters:
            - method:"z-score","min_max",
        """
        self.method = method
        self.histogram_bins = histogram_bins
        self.student_key = student_key
        self.teacher_key = teacher_key
        self.diff = diff
    def __call__(self, sample):
        for key in sample.keys():
            if key == 'label':
                continue
            if self.method == "z-score":
                sample[key] = self.z_score_normalization(sample[key])
            elif self.method == "min_max":
                sample[key] = self.min_max_normalization(sample[key])
            else :
                raise Exception(f"normalize method error: not exist{self.method}")
            return sample
        
        
    # Z-score normalization
    def z_score_normalization(self, image):

        _mean = np.mean(image)
        _std = np.std(image)
        z_score_normalized_image = (image - _mean) / (max(_std, 1e-8))
        #print(np.max(z_score_normalized_image))
        return z_score_normalized_image

    def min_max_normalization(self, image, max_val=None,min_val=None):
        if max_val is None:
            max_val = image.max()
        if min_val is  None:
            min_val = 0

        image[image< min_val] = min_val
        image[image> max_val] = max_val
        min_max_normalized_image = (1 * (image - min_val) / (max_val - min_val)) + 0

        return min_max_normalized_image


class Sample_Random_Cutout:
    def __init__(self, max_cutout_cube_size=32, min_cutout_cube_size = 8, percentage=0.15,fill ="min") -> None:
        self.max_cutout_cube_size = max_cutout_cube_size
        self.min_cutout_cube_size = min_cutout_cube_size
        self.percentage = percentage
        self.fill = fill
    def __call__(self, sample):
        assert 'label' in sample, 'missing label'
        if(random.random() < self.percentage and not np.max(sample['label']) == 0):
 
            return self.random_cutout(sample)
        else:
            return sample


    def random_cutout(self, sample):
        z, h, w = sample['label'].shape
        
        top_z = np.random.randint(0, z - self.max_cutout_cube_size)
        top_y = np.random.randint(0, h - self.max_cutout_cube_size)
        top_x = np.random.randint(0, w - self.max_cutout_cube_size)
        
        bottom_z = top_z + self.max_cutout_cube_size
        bottom_y = top_y + self.max_cutout_cube_size
        bottom_x = top_x + self.max_cutout_cube_size
        
        cutout_mask = np.ones(sample['label'].shape)
        cutout_mask[top_z:bottom_z, top_y:bottom_y, top_x:bottom_x] = 0
        cut = cutout_mask==0
        for key in sample.keys():
            if key == 'label':
                sample['label'][cut] = 0
            else:
                sample[key][cut] = sample[key].min() if self.fill =="min" else 0  

        return sample

class Sample_Adjust_contrast():
    def __init__(self, mode = "weak", percentage = 0.15) -> None:

        self.mode = mode
        self.do_contrast_percentage = percentage

        if mode == "strong":
            self.factor_range = (0.5, 1.5)
        else :
            self.factor_range = (0.75, 1.25)
    def __call__(self, sample):
        return self.adjust(sample)

    def adjust(self, sample, factor_range=(0.75,1.25)):
        if(random.random() >  self.do_contrast_percentage):
            return sample
        if((random.random() < 0.5) and (factor_range[0] < 1)):
            factor = np.random.uniform(factor_range[0], 1)
        else:
            factor = np.random.uniform(max(factor_range[0], 1), factor_range[1])
        for key in sample.keys():
            if key == 'label':
                continue
            _mean = sample[key].mean()
            _max = sample[key].max()
            _min = sample[key].min()
            sample[key] = (sample[key] - _mean)*factor + _mean
            sample[key][sample[key] < _min] = _min
            sample[key][sample[key] > _max] = _max

        #print("do adjust contrast")
        return sample
class Sample_Add_Noise:
    def __init__(self, percentage=0.1, varaince_range = (0, 0.1)) -> None:
        self.percentage = percentage
        self.varaince_range = varaince_range
    def __call__(self, sample):

        return self.add_noise(sample)

    def add_noise(self, sample):
        varaince = np.random.uniform(self.varaince_range[0], self.varaince_range[1])   
        if(random.random() < self.percentage):
            #print("do add_noise")
            size = sample[next(iter(sample.keys()))].shape
            noise = np.random.normal(0, varaince, size=size)
            for key in sample.keys():
                if key == 'label':
                    continue
                sample[key] = sample[key] + noise
        return sample
class Sample_Gussian_Blur:
    def __init__(self, percentage=0.2, sigma_range = (0.5, 1)) -> None:
        self.percentage = percentage
        self.sigma_range = sigma_range
    def __call__(self, sample):
        return self.gussian_blur(sample)

    def gussian_blur(self, sample):
        if( random.random() < self.percentage):        
            self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])   
            #print("do gussian_blur")
            for key in sample.keys():
                if key == 'label':
                    continue
                sample[key] = ndi.gaussian_filter(sample[key], sigma=self.sigma)
        return sample
class Sample_Brightness_Multiply:
    def __init__(self, percentage=0.15, multiply_range = (0.75, 1.25)) -> None:
        self.percentage = percentage
        self.multiply_range = multiply_range
    def __call__(self, sample):
        return self.brightness_multiply(sample)
    def brightness_multiply(self, sample):
        if( random.random() < self.percentage):        
            multiply = np.random.uniform(self.multiply_range[0], self.multiply_range[1]) 
            for key in sample.keys():
                if key == 'label':
                    continue
                sample[key] *= multiply
        return sample
class SampleLowRes():
    def __init__(self, low_res_range=(0.5,1), low_res_percentage = 0.25) -> None:
        self.low_res_range = low_res_range
        self.low_res_percentage = low_res_percentage
    def __call__(self, sample):


        if(random.random()<self.low_res_percentage):
            scale_factor = np.random.uniform(self.low_res_range[0], self.low_res_range[1])
            size = sample[next(iter(sample.keys()))].shape
            down_size = (int(size[0]*scale_factor),int(size[1]*scale_factor),int(size[2]*scale_factor))
            downsample = mt.Resize(down_size,mode="nearest")
            for key in sample.keys():
                if key == 'label':
                    continue
                low_res = downsample(sample[key].reshape(1, *size))
                upsample = mt.Resize(sample[key].shape,mode="trilinear")
                sample[key] = upsample(low_res).squeeze(0).numpy()

            return sample
        else:
            return sample      
class SampleGama():
    def __init__(self, gamma_range=(0.7,1.5), invert_image=False, retain_stats=True, percentage = 0.1, epsilon=1e-7) -> None:
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.percentage = percentage
        self.retain_stats = retain_stats
        self.epsilon = epsilon

    def __call__(self, sample):

        image_depth, image_height, image_width = sample[next(iter(sample.keys()))].shape

        if(random.random() > self.percentage):
            return sample
        else:            
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
            for key in sample.keys():
                if key == 'label':
                    continue
                if self.invert_image:
                    sample[key] = - sample[key]
                if self.retain_stats:
                    mn = sample[key].mean()
                    sd = sample[key].std()

                minm = sample[key].min()
                rnge = sample[key].max() - minm
                sample[key] = np.power(((sample[key] - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm
                if self.retain_stats:
                    sample[key] = sample[key] - sample[key].mean()
                    sample[key] = sample[key] / (sample[key].std() + 1e-8) * sd
                    sample[key] = sample[key] + mn
                if self.invert_image:
                    sample[key] = - sample[key]

            return sample
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        for key in sample.keys():
            #print(key)
            if key == "label":
                sample['label'] = torch.from_numpy(sample['label']).long()
            else:
                sample[key] = sample[key].reshape(1, *(sample[key].shape)).astype(np.float32)
                sample[key] = torch.from_numpy(sample[key])


        return sample

          
    
def test():
    from torchvision import transforms
    import SimpleITK as sitk
    import os
    import pickle
    # np.random.seed(config.args.seed)
    # random.seed(config.args.seed)    
    # torch.manual_seed(config.args.seed)
    img_stik =sitk.ReadImage(config.test_img_path)
    image = sitk.GetArrayFromImage(img_stik)
    label = sitk.GetArrayFromImage(sitk.ReadImage(config.test_label_path))
    patch_size = (96, 160, 160)
    import time
    cb = CurrentLabelBatch(2)

    print("finish")
    tumor_list = []
    for f in os.listdir("/media/wl/Extreme SSD/my_net/data/CA/tumors"):
        if f.endswith(".nii.gz"):
            path = os.path.join("/media/wl/Extreme SSD/my_net/data/CA/tumors",f)
            tumor_list.append(sitk.GetArrayFromImage(sitk.ReadImage(path)))
    s = time.time()

    data = {}


    data_trans = transforms.Compose([ 
                        Sample_Normalize(),
                        SampleRandomRotateZ(30),
                        SampleRandomScale(),
                        SampleRandomCrop(patch_size,current_label_batch=cb),                          
                        Sample_Add_Noise(percentage=1),
                        Sample_Gussian_Blur(percentage=1),
                        Sample_Brightness_Multiply(),                            
                        SampleRandomFlip(),                              
                        SampleLowRes(low_res_percentage=1),
                        Sample_Random_Cutout(), 
                        ])

    data['image'] = image
    data['label'] = label
    data = data_trans(data)

    e = time.time()
    print("data transform time per sample:", e-s)
    if(not os.path.exists("test")):
        os.makedirs("test")
    img = sitk.GetImageFromArray( data['image'])
    img.CopyInformation(img_stik)
    l =sitk.GetImageFromArray(data['label'])
    l.CopyInformation(img_stik)
    sitk.WriteImage(img,"semi/test/trans_img.nii.gz")
    sitk.WriteImage(l,"semi/test/trans_label.nii.gz")

if __name__ == "__main__":
    test()

