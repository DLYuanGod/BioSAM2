# %% import packages
import numpy as np
import os
from skimage import io, transform, measure
from tqdm import tqdm

# convert image to npz files, including original image and corresponding masks
modality = "CT"
anatomy = "Cell"  # anatomy + dataset name
img_name_suffix = "_0000.png"  # 修改为图像文件的后缀
gt_name_suffix = ".png"  # ground truth 文件的后缀
prefix = modality + "_" + anatomy + "_"

img_path = "/root/autodl-tmp/nnUNet_raw/Dataset703_NeurIPSCell/imagesTr"  # 修改为图像文件的路径
gt_path = "/root/autodl-tmp/nnUNet_raw/Dataset703_NeurIPSCell/labelsTr"  # 修改为ground truth文件的路径
npy_path = "/root/autodl-tmp/nnUNet_raw/Dataset703_NeurIPSCell/npy/" + prefix[:-1]
os.makedirs(os.path.join(npy_path, "gts"), exist_ok=True)
os.makedirs(os.path.join(npy_path, "imgs"), exist_ok=True)

image_size = 1024
pixel_num_thre2d = 100

names = sorted(os.listdir(gt_path))
print(f"ori # files len(names)={len(names)}")
names = [
    name
    for name in names
    if os.path.exists(os.path.join(img_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check # files len(names)={len(names)}")

# Debugging: Print the first few file names
print("First few valid files:", names[:10])

# set label ids that are excluded
remove_label_ids = [
    12
]  # remove duodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks

# %% save preprocessed images and masks as npz files
for name in tqdm(names):  # process all files
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    
    # Debugging: Print the file paths being processed
    print(f"Processing image: {os.path.join(img_path, image_name)}, ground truth: {os.path.join(gt_path, gt_name)}")
    
    gt_data_ori = io.imread(os.path.join(gt_path, gt_name))
    # remove label ids
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # label tumor masks as instances and remove from gt_data_ori
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        # label tumor masks as instances
        tumor_inst = measure.label(tumor_bw, connectivity=2)
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # remove small objects with less than 100 pixels in 2D
    labeled_gt, num_features = measure.label(gt_data_ori, return_num=True, connectivity=2)
    for region in measure.regionprops(labeled_gt):
        if region.area < pixel_num_thre2d:
            for coordinates in region.coords:
                gt_data_ori[coordinates[0], coordinates[1]] = 0

    # load image and preprocess
    image_data = io.imread(os.path.join(img_path, image_name))
    if modality == "CT":
        # normalize CT images
        WINDOW_LEVEL = 40
        WINDOW_WIDTH = 400
        lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
        upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
    else:
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0

    image_data_pre = np.uint8(image_data_pre)
    np.savez_compressed(os.path.join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=image_data_pre, gts=gt_data_ori)

    # resize and save each image and ground truth as npy files
    resize_img_skimg = transform.resize(
        image_data_pre,
        (image_size, image_size),
        order=3,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True,
    )
    resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
        resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W)
    resize_gt_skimg = transform.resize(
        gt_data_ori,
        (image_size, image_size),
        order=0,
        preserve_range=True,
        mode="constant",
        anti_aliasing=False,
    )
    resize_gt_skimg = np.uint8(resize_gt_skimg)
    assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape
    np.save(
        os.path.join(
            npy_path,
            "imgs",
            prefix
            + gt_name.split(gt_name_suffix)[0]
            + ".npy",
        ),
        resize_img_skimg_01,
    )
    np.save(
        os.path.join(
            npy_path,
            "gts",
            prefix
            + gt_name.split(gt_name_suffix)[0]
            + ".npy",
        ),
        resize_gt_skimg,
    )
