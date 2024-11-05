import os
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from typing import Tuple, List, Union
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndi
from net import Net



def z_score_normalization(image):

    _mean = np.mean(image)
    _std = np.std(image)
    z_score_normalized_image = (image - _mean) / (max(_std, 1e-8))
    return z_score_normalized_image
def min_max_normalization(image, max_val=None,min_val=None):
    if max_val is None:
        max_val = image.max()
    if min_val is  None:
        min_val = 0

    image[image< min_val] = min_val
    image[image> max_val] = max_val
    min_max_normalized_image = (1 * (image - min_val) / (max_val - min_val)) + 0

    return min_max_normalized_image


def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8, dtype=np.float16) \
        -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1 
    # 将中心位置设置为 1，以便在后续生成高斯核时保证高斯峰位于中心。
    # tmp 是中心为 1 的数组，sigmas 是每个维度的标准差，mode 参数设置为 'constant' 表示在边界外填充常数值，cval 为填充的常数值。
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    # 将权重图中为 0 的像素值设置为非零像素值的最小值，以防止出现 NaN（不是一个数字）。
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    """
    - Returns (List[List[int]]): z, y, x上的步长, shape like (3, step_num_of_each_axis
    )
    """
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    if isinstance(tile_step_size,list):
        for t in tile_step_size:
            assert 0 < t <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
    else:
        assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
        t = [tile_step_size]*3
        tile_step_size = t

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * t for i,t in zip(tile_size,tile_step_size)]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

def pad_nd_image(image: Union[torch.Tensor, np.ndarray], new_shape: Tuple[int, ...] = None,
                 mode: str = "constant", kwargs: dict = None, return_slicer: bool = False,
                 shape_must_be_divisible_by: Union[int, Tuple[int, ...], List[int]] = None) -> \
        Union[Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor, np.ndarray], Tuple]]:
    """
    One padder to pad them all. Documentation? Well okay. A little bit

    Padding is done such that the original content will be at the center of the padded image. If the amount of padding
    needed it odd, the padding 'above' the content is larger,
    Example:
    old shape: [ 3 34 55  3]
    new_shape: [3, 34, 96, 64]
    amount of padding (low, high for each axis): [[0, 0], [0, 0], [20, 21], [30, 31]]

    :param image: can either be a numpy array or a torch.Tensor. pad_nd_image uses np.pad for the former and
           torch.nn.functional.pad for the latter
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
           len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in
           any of the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)

           Example:
           image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
           image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: will be passed to either np.pad or torch.nn.functional.pad depending on what the image is. Read the
           respective documentation!
    :param return_slicer: if True then this function will also return a tuple of python slice objects that you can use
           to crop back to the original image (reverse padding)
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
           divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match
           that will be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation (numpy) or torch.nn.functional.pad (torch)

    :returns: if return_slicer=False, this function returns the padded numpy array / torch Tensor. If
              return_slicer=True it will also return a tuple of slice objects that you can use to revert the padding:
              output, slicer = pad_nd_image(input_array, new_shape=XXX, return_slicer=True)
              reversed_padding = output[slicer] ## this is now the same as input_array, padding was reversed
    """
    if kwargs is None:
        kwargs = {}

    old_shape = np.array(image.shape)

    if shape_must_be_divisible_by is not None:
        assert isinstance(shape_must_be_divisible_by, (int, list, tuple, np.ndarray))
        if isinstance(shape_must_be_divisible_by, int):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(image.shape)
        else:
            if len(shape_must_be_divisible_by) < len(image.shape):
                shape_must_be_divisible_by = [1] * (len(image.shape) - len(shape_must_be_divisible_by)) + \
                                             list(shape_must_be_divisible_by)

    if new_shape is None:
        assert shape_must_be_divisible_by is not None
        new_shape = image.shape

    if len(new_shape) < len(image.shape):
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)

        if len(shape_must_be_divisible_by) < len(new_shape):
            shape_must_be_divisible_by = [1] * (len(new_shape) - len(shape_must_be_divisible_by)) + \
                                         list(shape_must_be_divisible_by)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] %
                              shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        if isinstance(image, np.ndarray):
            res = np.pad(image, pad_list, mode, **kwargs)
        elif isinstance(image, torch.Tensor):
            # torch padding has the weirdest interface ever. Like wtf? Y u no read numpy documentation? So much easier
            torch_pad_list = [i for j in pad_list for i in j[::-1]][::-1]
            import torch.nn.functional as F
            res = F.pad(image, torch_pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer

def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float,
                                 verbose: bool = False):
    """
    切片生成逻辑：

    针对每个维度，使用 compute_steps_for_sliding_window 函数
        计算在该维度上滑动窗口时需要的步长（stride）。
        使用这些步长生成切片，并将它们添加到 slicer 中
    """
    # image_size:3---z, y, x
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) ' \
                                                      'must be one shorter than len(image_size) (only dimension ' \
                                                      'discrepancy of 1 allowed).'
        # tile_size缺少一个维度
        # 说明是3d输入2d网络
        # self.tile_step_size=0.5, 每次移动窗口的一半
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
        if verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
        if np.prod([len(i) for i in steps])>20:
            tile_step_size = [1,1,0.5]
            steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
        # if np.prod([len(i) for i in steps])>20:
        #     tile_step_size = [1,1,1]
        #     steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)

        if verbose: print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    yield slicer
def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None) \
        -> torch.Tensor:
    """
    - Returns (torch.Tensor): 所有镜像输入的预测结果的均值
    """

    prediction = network(x)
    # print(np.unique(prediction.cpu().numpy()))
    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons
    return prediction
def predict_sliding_window_return_logits(network: nn.Module,
                                         input_image: torch.Tensor,
                                         num_segmentation_heads: int,
                                         tile_size: Tuple[int, ...],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         precomputed_gaussian: torch.Tensor = None,
                                         verbose: bool = True,
                                         device: torch.device = torch.device('cuda')) -> Union[np.ndarray, torch.Tensor]:

    network = network.to(device)
    network.eval()

    with torch.no_grad():
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

        results_device = torch.device('cpu')

        if verbose: print("step_size:", tile_step_size)
        if verbose: print("mirror_axes:", mirror_axes)

        # if input_image is smaller than tile_size we need to pad it to tile_size.
        # 输入图片小于tile_size则需要填充成tile_size
        data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True, None)

        if use_gaussian:
            # 在滑动窗口预测过程中，
            # 代码会在每个窗口中将预测结果乘以对应位置的高斯核值，从而实现在窗口内进行加权平均。
            gaussian = torch.from_numpy(
                compute_gaussian(tile_size, sigma_scale=1. / 8)) if precomputed_gaussian is None else precomputed_gaussian
            gaussian = gaussian.half()
            # make sure nothing is rounded to zero or we get division by zero :-(
            mn = gaussian.min()
            if mn == 0:
                gaussian.clip_(min=mn)
        # data.shape[1:]: 除去通道数以外的维度形状(z_shape, y_shape, x_shape)
        slicers = get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size, verbose=verbose)

        # preallocate results and num_predictions. Move everything to the correct device
        try:
            predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                            device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                        device=results_device)
            gaussian = gaussian.to(results_device)
        except RuntimeError:
            # sometimes the stuff is too large for GPUs. In that case fall back to CPU
            results_device = torch.device('cpu')
            predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                            device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                        device=results_device)
            gaussian = gaussian.to(results_device)

        for sl in slicers:
            # (1, c, z, x, y)
            workon = data[sl][None]
            workon = workon.to(device, non_blocking=False)
            # (num_class, z, x, y)
            prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[0].to(results_device)
            # (num_class, z, x, y)
            predicted_logits[sl] += (prediction * gaussian if use_gaussian else prediction)
            # 如果 self.use_gaussian 为真，将高斯核 gaussian 加到 n_predictions 上
            # n_predictions = ∑gaussian_i
            # (z, x, y)
            n_predictions[sl[1:]] += (gaussian if use_gaussian else 1)
        # 进行加权求和 ∑（pred_i*gaussian_i）/∑gaussian_i
        predicted_logits /= n_predictions
    # slicer_revert_padding[1:]: 这部分索引表示从 slicer_revert_padding 中除了第一个元素外的其余部分。
    # 如果原图小于tile_size, 进行过填充，则去掉填充
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]





def resample(image_sitk, is_label = False, new_spacing=(1,1,1))->None:

    img = image_sitk
    resample = sitk.ResampleImageFilter()
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
    return img
def resample_size(image_sitk, new_size,new_spacing, is_label = False)->None:

    img = image_sitk
    resample = sitk.ResampleImageFilter()
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
    return img
def get_volume_orientation(volume: sitk.Image) -> str:
    orient_filter = sitk.DICOMOrientImageFilter()
    cosines = volume.GetDirection()
    return orient_filter.GetOrientationFromDirectionCosines(cosines)
def reorient_to_lps(volume: sitk.Image,
                    ) -> sitk.Image:
    """
    If volume is not in the LPS orientation, reorient to LPS
    """
    current_orientation = get_volume_orientation(volume)
    if current_orientation != "LPS":
        orientation_filter = sitk.DICOMOrientImageFilter()
        orientation_filter.SetDesiredCoordinateOrientation("LPS")
        volume = orientation_filter.Execute(volume)
    return volume

def reorient_to_origin(volume: sitk.Image,ori) -> sitk.Image:
    """
    If volume is not in the LPS orientation, reorient to LPS
    """
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(ori)
    volume = orientation_filter.Execute(volume)
    return volume


def predict( img_path,model, out_dir, img_suffix, out_suffix, num_class):
    img_name = os.path.basename(img_path)
    image_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
    origin_orient = get_volume_orientation(image_sitk)
    image_sitk_lps = reorient_to_lps(image_sitk)
    origin_sapcing = image_sitk_lps.GetSpacing()
    orign_size = image_sitk_lps.GetSize()
    resample_image_sitk = resample(image_sitk_lps,False,(1.2,1.2,3))
    resample_image = sitk.GetArrayFromImage(resample_image_sitk)
    z, y, x = resample_image.shape
    
    resize_normalize_image = z_score_normalization(resample_image)#min_max_normalization(resample_image,240,-160)#
    input = torch.from_numpy(resize_normalize_image.reshape(
            1, resize_normalize_image.shape[0], resize_normalize_image.shape[1], resize_normalize_image.shape[2]).astype(np.float32)
        ).to('cuda')

    resample_pred = predict_sliding_window_return_logits(network=model, input_image=input, 
                                                num_segmentation_heads=num_class
                                                , tile_size=[48, 160, 224]
                                                ,mirror_axes=([0,1,2])
                                                )
    num_class, z, y, x = resample_pred.shape
    resample_pred = torch.softmax(resample_pred.float(),dim=0)
    resample_pred = torch.argmax(resample_pred, dim=0).numpy()
    if z > 80:
        resample_pred = post_process_all_body(resample_pred)
    resample_pred_sitk = sitk.GetImageFromArray(resample_pred)
    resample_pred_sitk.CopyInformation(resample_image_sitk)
    pred_sitk = resample_size(resample_pred_sitk,orign_size,origin_sapcing,True)
    pred_sitk_original = reorient_to_origin(pred_sitk, origin_orient)
    out_path = os.path.join(out_dir, img_name.replace(img_suffix, out_suffix))
    sitk.WriteImage(pred_sitk_original, out_path)

LABELS= {
        "background": 0,
        "Liver": 1,
        "Right Kidney": 2,
        "Right Spleen": 3,
        "Pancreas": 4,
        "Aorta": 5,
        "Inferior Vena Cava": 6,
        "Right Adrenal Gland": 7,
        "Left Adrenal Gland": 8,
        "Gall Bladder": 9,
        "Esophagus": 10,
        "Stomach": 11,
        "Duodenum": 12,
        "Left Kidney": 13
        }

def post_process_all_body(label):
    liver = label ==LABELS['Liver']
    l = max_region(liver)
    z_livers,_,_ = np.where(l==1)
    if len(z_livers)==0:
        return  label
    z_liver = np.mean(z_livers)
    _max = min(label.shape[0],np.max(z_livers)+8)
    _min = max(0,np.min(z_livers)-8)
    liver_box = np.zeros(label.shape)
    liver_box[_min:_max] = 1
    label[(label==LABELS["Liver"])*(liver_box==0)] = 0
    label[(label==LABELS["Stomach"])*(liver_box==0)] = 0
    label[(label==LABELS["Gall Bladder"])*(liver_box==0)] = 0
    label[(label==LABELS["Pancreas"])*(liver_box==0)] = 0
    label[(label==LABELS["Left Adrenal Gland"])*(liver_box==0)] = 0
    label[(label==LABELS["Right Adrenal Gland"])*(liver_box==0)] = 0




    a = max_region(label==LABELS["Aorta"],liver_box)
    label[(label==LABELS["Aorta"])] = 0
    label[a==1] = LABELS["Aorta"]

    lk = max_region((label==LABELS["Left Kidney"]))
    label[(label==LABELS["Left Kidney"])] = 0
    label[lk==1] = LABELS["Left Kidney"]
    
    rk = max_region((label==LABELS["Right Kidney"]))
    label[(label==LABELS["Right Kidney"])] = 0
    label[rk==1] = LABELS["Right Kidney"]

    z_lks,_,_ = np.where(lk==1)
    if len(z_lks)==0:
        return  label
    z_lk = np.mean(z_lks)
    if z_lk > z_liver:
        z_max = np.max(z_lks)
        z_liver_min = np.min(z_livers)
        z_min = max(z_liver_min-10,0)
        liver2kidney_box = np.zeros(label.shape)
        liver2kidney_box[z_liver_min:,:,:]=1
        label[(label==LABELS["Duodenum"])*(liver2kidney_box==0)] = 0
        label[(label==LABELS["Pancreas"])*(liver2kidney_box==0)] = 0

    else:
        z_liver_max = np.max(z_livers)
        z_max = min(z_liver_max + 10,label.shape[0])
        z_min = max(np.min(z_lks),0)
        liver2kidney_box = np.zeros(label.shape)
        liver2kidney_box[:z_liver_max,:,:]=1
        label[(label==LABELS["Duodenum"])*(liver2kidney_box==0)] = 0
        label[(label==LABELS["Pancreas"])*(liver2kidney_box==0)] = 0
    spleen_box = np.zeros(label.shape)
    spleen_box[z_min:z_max] = 1
    label[(label==LABELS["Right Spleen"])*(spleen_box==0)] = 0
    

    return label


    
def max_region(_region,box=None,keep_second=False):
    log = ""
    _region = _region.astype(np.uint8)
    if(np.all(_region == 0)):
        return _region
    # 标记连通域
    labeled_image, num_features = ndi.label(_region)
    if box is None:
        # 计算各个连通域的大小
        sizes = ndi.sum(_region, labeled_image, range(num_features + 1))
    else:
        sizes = ndi.sum(_region, labeled_image*box, range(num_features + 1))
    sorted_indices = np.argsort(sizes[1:])[::-1]        
    # top1_size = sizes[sorted_indices[0]+1]
    # 找到最大的连通域
    max_label = sorted_indices[0] + 1  # 从1开始是因为0是背景标签
    # log = log + f"\n\tregion_1_size:{top1_size}"
    max_region = np.zeros_like(_region)
    max_region[labeled_image == max_label] = 1
    if keep_second and len(sorted_indices)>1:
        s_label = sorted_indices[1] + 1  # 从1开始是因为0是背景标签
        max_region[labeled_image == s_label] = 1
    return max_region

def main(model_path,image_folder,predict_folder,in_sub):

    num_class =14
    model = Net(n_channels=1, n_classes=num_class,normalization="instancenorm")
    model = model.cuda()
    pretrained_weights = torch.load(model_path)
    model.load_state_dict(pretrained_weights)
    model.has_dropout = False
    model.deep_supervision = False
    model.eval()
    img_folder = image_folder
    pred_folder = predict_folder
    print(img_folder)
    if(not os.path.exists(pred_folder)):
        os.makedirs(pred_folder)
    exist_pred_list = os.listdir(pred_folder)
    files = [f for f in os.listdir(img_folder) if f.endswith(in_sub) and  str.replace(f,"_0000","") not in exist_pred_list]
    print(str(len(files))+" files left.")
    print(files)
    import time
    for i in files:
        print(f"********************predict {i}********************")
        s = time.time()
        predict( os.path.join(img_folder, i), model,pred_folder, "_0000.nii.gz", ".nii.gz", 
                num_class=num_class)
        print(f"time: {time.time() - s}")


    

if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--model_path', type=str, default='./model/Mean_Teacher_f1/semi_model/best_model.pth')
    parser.add_argument('--image_folder', type=str, default='./STD/MRI/LLD')  
    parser.add_argument('--predict_folder', type=str, default='./data/stage1_pred_t1w')    
    parser.add_argument('--in_sub', type=str, default='_C-pre_0000.nii.gz')  


    args = parser.parse_args()
    main(args.model_path,args.image_folder,args.predict_folder,args.in_sub)
