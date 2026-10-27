from re import L
import numpy as np
from monai import transforms



if not hasattr(transforms, "AddChanneld"):
    def _compat_AddChanneld(keys, allow_missing_keys=False, **kwargs):
        return transforms.EnsureChannelFirstd(
            keys=keys,
            channel_dim="no_channel",
            allow_missing_keys=allow_missing_keys,
        )
    transforms.AddChanneld = _compat_AddChanneld

from torch import scalar_tensor, zero_

class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):









    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            result = []
            result.append(np.logical_or(label == 1, label == 4))
            result.append(np.logical_or(np.logical_or(label == 1, label == 2), label == 4))
            result.append(label == 4)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d

def get_scratch_train_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.RandScaleIntensityd(keys="image",
                                            factors=0.1,
                                            prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image",
                                            offsets=0.1,
                                            prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),

                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=getattr(args, 'brats_pos', 1), neg=getattr(args, 'brats_neg', 1), num_samples=args.num_samples,
                    image_key="image", image_threshold=0),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),

                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"])

            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_mae_pretrain_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),

                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=getattr(args, 'brats_pos', 1), neg=getattr(args, 'brats_neg', 1), num_samples=args.num_samples,
                    image_key="image", image_threshold=0),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),

                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_val_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

def get_vis_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),










                transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),











                transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

def get_raw_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

class Resize():
    def __init__(self, scale_params):
        self.scale_params = scale_params

    def __call__(self, img):
        scale_params = self.scale_params
        shape = img.shape[1:]
        assert len(scale_params) == len(shape)
        spatial_size = []
        for scale, shape_dim in zip(scale_params, shape):
            spatial_size.append(int(scale * shape_dim))
        transform = transforms.Resize(spatial_size=spatial_size, mode='nearest')


        return transform(img)

def get_post_transforms(args):
    if args.dataset == 'btcv':
        if args.test:
            post_pred = transforms.Compose([transforms.EnsureType(),

                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),

                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
    elif args.dataset == 'msd_brats':
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)])
        post_label = transforms.Identity()
    return post_pred, post_label





_ORIG_get_mae_pretrain_transforms = get_mae_pretrain_transforms
_ORIG_get_vis_transforms = get_vis_transforms

def get_mae_pretrain_transforms(args):
    if getattr(args, "dataset", None) == "fomo50k":
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.RandSpatialCropd(
                    keys=["image"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    random_size=False,
                ),
                transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        return train_transform

    return _ORIG_get_mae_pretrain_transforms(args)

def get_vis_transforms(args):
    if getattr(args, "dataset", None) == "fomo50k":
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.CenterSpatialCropd(
                    keys=["image"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                ),
                transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        return val_transform

    return _ORIG_get_vis_transforms(args)







_PRE_ALIGN_get_val_transforms = get_val_transforms
_PRE_ALIGN_get_vis_transforms = get_vis_transforms

def get_val_transforms(args):
    if getattr(args, "dataset", None) == "msd_brats":
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        return val_transform

    return _PRE_ALIGN_get_val_transforms(args)


def get_vis_transforms(args):
    if getattr(args, "dataset", None) == "msd_brats":
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        return val_transform

    return _PRE_ALIGN_get_vis_transforms(args)





class ChannelWiseThreshold:
    def __init__(self, thresholds):
        if isinstance(thresholds, str):
            thresholds = thresholds.replace("[", "").replace("]", "").split(",")
            thresholds = [float(x.strip()) for x in thresholds if x.strip()]
        self.thresholds = thresholds

    def __call__(self, x):
        import torch
        th = torch.as_tensor(self.thresholds, dtype=x.dtype, device=x.device)

        if x.ndim == 5:
            th = th.view(1, -1, 1, 1, 1)
        elif x.ndim == 4:
            th = th.view(-1, 1, 1, 1)
        else:
            raise RuntimeError(f"Unsupported prediction shape for ChannelWiseThreshold: {tuple(x.shape)}")

        return (x > th).float()


_PRE_CHANNEL_THRESH_get_post_transforms = get_post_transforms

def get_post_transforms(args):
    if getattr(args, "dataset", None) == "msd_brats":
        thresholds = getattr(args, "brats_thresholds", [0.4, 0.6, 0.5])
        post_pred = transforms.Compose([
            transforms.EnsureType(),
            transforms.Activations(sigmoid=True),
            ChannelWiseThreshold(thresholds),
        ])
        post_label = transforms.Identity()
        return post_pred, post_label

    return _PRE_CHANNEL_THRESH_get_post_transforms(args)
