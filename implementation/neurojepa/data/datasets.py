import sys
import logging
import warnings
import numpy as np
import pandas as pd

from monai import data
from monai.utils.type_conversion import convert_data_type

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from neurojepa.data.transforms import loading_transforms
from neurojepa.masks.masking import MaskCollator

from typing import List, Tuple, Dict, Any, Optional

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def custom_collate_fn(batch: List[Any]) -> Any:



    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)





class BaseMRIDataset(Dataset):










    def __init__(
        self,
        img_size: List[int],
        model_name: str,
        in_chans: int,
        csv_file: str,
        data_augmentation: Any,
        dataframe: Optional[pd.DataFrame] = None,
        cache_dir: Optional[str] = None,
        img_col: str = "img_path",
        img_flag: str = "img_qc",
        dropna: bool = False,
    ):
        self.img_size = img_size
        self.model_name = model_name
        self.in_chans = in_chans
        self.img_col = img_col
        self.img_flag = img_flag
        self.dropna = dropna


        if not any(tag in self.model_name for tag in ("jepa", "vit", "voco", "cnn", "brainIAC", "3dino")):
            raise ValueError(f"Unsupported model name: {self.model_name}")


        if dataframe is None:
            df = pd.read_csv(csv_file)
        else:
            df = dataframe

        if self.img_col not in df.columns:
            raise ValueError(f"CSV must contain an '{self.img_col}' column.")

        if self.img_flag in df.columns:
            df = df[df[self.img_flag] == True].reset_index(drop=True)
            print(f"Dropped rows with bad QC in '{self.img_flag}'. New dataset size: {len(df)}")

        cache_augmentation = loading_transforms(roi=self.img_size, model_name=self.model_name)
        self.data_augmentation = data_augmentation
        self.cache_dir = cache_dir
        self._paths = df[self.img_col].astype(str).tolist()
        self._df = df


        self._cache_ds = data.PersistentDataset(
            data=[{"image": p} for p in self._paths],
            transform=cache_augmentation,
            cache_dir=self.cache_dir,
        )


        self.placeholder_image = self._build_placeholder(self.img_size, self.in_chans)

    @staticmethod
    def _build_placeholder(img_size: Tuple[int, ...], in_chans: int) -> torch.Tensor:



        ph = torch.zeros((in_chans, *img_size), dtype=torch.float32)
        meta_tensor, *_ = convert_data_type(ph, output_type=data.MetaTensor)
        return meta_tensor

    def _valid_sample(self, sample: Dict[str, Any], idx: int) -> bool:






        validity = True

        if "image" not in sample:
            warnings.warn(f"[idx={idx}] Missing 'image' key for file {self._paths[idx]}.")
            validity = False
        else:
            img = sample["image"]
            if not isinstance(img, torch.Tensor):
                warnings.warn(f"[idx={idx}] Image is not a tensor (got {type(img)}) for file {self._paths[idx]}.")
                validity = False
            else:
                if img.ndim != (1 + len(self.img_size)):
                    warnings.warn(f"[idx={idx}] Unexpected dims {tuple(img.shape)}; expected [C, *img_size] for file {self._paths[idx]}.")
                    validity = False
                if img.shape[0] != self.in_chans:
                    warnings.warn(f"[idx={idx}] Channel mismatch: got {img.shape[0]} vs {self.in_chans} for file {self._paths[idx]}.")
                    validity = False

        return validity

    def __len__(self) -> int:
        return len(self._paths)

    def _fallback_image(self) -> torch.Tensor:
        return self.placeholder_image





class PretrainDataset(BaseMRIDataset):






    def __getitem__(self, idx: int):
        try:
            sample = self._cache_ds[idx]
            if not self._valid_sample(sample, idx):
                return self._fallback_image()

            sample = self.data_augmentation(sample)
            image = sample["image"] if isinstance(sample, dict) else sample
            modality = self._df['modality_flag'].iloc[idx] if 'modality_flag' in self._df.columns else None

            if isinstance(modality, str):
                modality = MOD_MAP.get(modality, -1)
            elif modality is None:
                modality = -1

            modality = torch.tensor(modality, dtype=torch.long)

            return image, modality
        except Exception as e:
            warnings.warn(f"[idx={idx}] Error during load/augment for file {self._paths[idx]}: {e}")
            return self._fallback_image(), torch.tensor(-1, dtype=torch.long)





class MonitorDataset(BaseMRIDataset):




















    def __init__(
        self,
        img_size: List[int],
        model_name: str,
        in_chans: int,
        csv_file: str,
        data_augmentation: Any,
        label_cols: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        img_col: str = "img_path",
        img_flag: str = "img_qc",
        label_dtype: torch.dtype = torch.float32,
        na_as_zero: bool = True,
        dropna: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            model_name=model_name,
            in_chans=in_chans,
            csv_file=csv_file,
            data_augmentation=data_augmentation,
            cache_dir=cache_dir,
            img_col=img_col,
            img_flag=img_flag,
            dropna=dropna,
        )


        if label_cols is None:

            candidate_cols = [c for c in self._df.columns if c != self.img_col]
        else:
            candidate_cols = label_cols

        if self.dropna:
            self._df = self._df.dropna(subset=[img_col, label_col]).reset_index(drop=True)
            print(f"Dropped rows with NaN in '{img_col}' and '{label_col}'. \
                New dataset size: {len(self._df)}")
            self._paths = self._df[self.img_col].astype(str).tolist()


        missing = [c for c in candidate_cols if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing label columns in CSV: {missing}")


        labels_df = self._df[candidate_cols].copy()
        if na_as_zero:
            labels_df = labels_df.fillna(0)

        for c in labels_df.columns:
            labels_df[c] = (labels_df[c].astype(float) > 0.5).astype(int)

        self.label_cols = list(labels_df.columns)
        self.class_names = self.label_cols
        self.label_arr = labels_df.values.astype(np.int64)
        self.num_labels = self.label_arr.shape[1]
        self.label_dtype = label_dtype


        positives = self.label_arr.sum(axis=0)
        zero_pos = [self.label_cols[i] for i, v in enumerate(positives) if v == 0]
        if zero_pos:
            warnings.warn(f"Classes with zero positives in CSV (cannot train/measure properly): {zero_pos}")

    def __getitem__(self, idx: int):
        try:
            sample = self._cache_ds[idx]
            if not self._valid_sample(sample, idx):

                target = torch.zeros(self.num_labels, dtype=self.label_dtype)
                return self._fallback_image(), target


            sample = self.data_augmentation(sample)
            image = sample["image"] if isinstance(sample, dict) else sample


            target = torch.from_numpy(self.label_arr[idx]).to(self.label_dtype)
            return image, target
        except Exception as e:
            warnings.warn(f"[idx={idx}] Error during load/augment for file {self._paths[idx]}: {e}")
            target = torch.zeros(self.num_labels, dtype=self.label_dtype)
            return self._fallback_image(), target





class FinetuneDataset(BaseMRIDataset):



















    def __init__(
        self,
        img_size: List[int],
        model_name: str,
        in_chans: int,
        csv_file: str,
        data_augmentation: Any,
        label_col: str,
        num_classes: int,
        dataframe: Optional[pd.DataFrame] = None,
        cache_dir: Optional[str] = None,
        img_col: str = "img_path",
        label_dtype: torch.dtype = torch.float32,
        na_as_zero: bool = True,
        dropna: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__(
            img_size=img_size,
            model_name=model_name,
            in_chans=in_chans,
            csv_file=csv_file,
            dataframe=dataframe,
            data_augmentation=data_augmentation,
            cache_dir=cache_dir,
            img_col=img_col,
            dropna=dropna,
        )

        if self.dropna:
            self._df = self._df.dropna(subset=[img_col, label_col]).reset_index(drop=True)
            print(f"Dropped rows with NaN in '{img_col}' and '{label_col}'. \
                New dataset size: {len(self._df)}")


            if 't1w' in img_col and 'flair_qc' in self._df.columns:
                self._df = self._df[self._df['t1w_qc'] == False]
            elif 't2w' in img_col and 'flair_qc' in self._df.columns:
                self._df = self._df[self._df['t2w_qc'] == False]
            elif 'flair' in img_col and 'flair_qc' in self._df.columns:
                self._df = self._df[self._df['flair_qc'] == False]
            else:
                pass

            self._paths = self._df[self.img_col].astype(str).tolist()

        if label_col not in self._df.columns:
            raise ValueError(f"Missing label column '{label_col}' in CSV.")

        labels_series = self._df[label_col].copy()
        if na_as_zero:
            labels_series = labels_series.fillna(0)


        if num_classes == 1:

            labels_series = labels_series.astype(float)
            if mean is not None and std is not None:
                labels_series = (labels_series - mean) / std


            self.label_arr = labels_series
            self.label_dtype = torch.float32
        elif num_classes == 2:

            self.label_arr = (labels_series.astype(float) > 0.5).astype(np.int64)
            self.label_dtype = torch.float32
        elif num_classes > 2:

            self.label_arr = labels_series.astype(np.int64)
            self.label_dtype = torch.long
        else:
            raise ValueError("num_classes must be >= 1")

        self.label_col = label_col
        self.num_classes = num_classes


        if num_classes > 1:
            class_counts = self.label_arr.value_counts()
            for i in range(num_classes):
                if i not in class_counts or class_counts[i] == 0:
                    warnings.warn(f"Class {i} has zero samples in the dataset.")

    def __getitem__(self, idx: int):
        try:
            sample = self._cache_ds[idx]
            if not self._valid_sample(sample, idx):

                return self._fallback_image(), torch.tensor(0, dtype=self.label_dtype)


            sample = self.data_augmentation(sample)
            image = sample["image"] if isinstance(sample, dict) else sample


            target_val = self.label_arr.iloc[idx]
            target = torch.tensor(target_val, dtype=self.label_dtype)
            return image, target
        except Exception as e:
            warnings.warn(f"[idx={idx}] Error during load/augment for file {self._paths[idx]}: {e}")
            return self._fallback_image(), torch.tensor(0, dtype=self.label_dtype)





class SurvivalDataset(BaseMRIDataset):



















    def __init__(
        self,
        img_size: List[int],
        model_name: str,
        in_chans: int,
        csv_file: str,
        data_augmentation: Any,
        time_col: str,
        event_col: str,
        dataframe: Optional[pd.DataFrame] = None,
        cache_dir: Optional[str] = None,
        img_col: str = "img_path",
        dropna: bool = True,
        normalize_time: bool = False,
        time_mean: Optional[float] = None,
        time_std: Optional[float] = None,
    ):
        super().__init__(
            img_size=img_size,
            model_name=model_name,
            in_chans=in_chans,
            csv_file=csv_file,
            dataframe=dataframe,
            data_augmentation=data_augmentation,
            cache_dir=cache_dir,
            img_col=img_col,
            dropna=dropna,
        )


        if time_col not in self._df.columns:
            raise ValueError(f"Missing time column '{time_col}' in CSV.")
        if event_col not in self._df.columns:
            raise ValueError(f"Missing event column '{event_col}' in CSV.")


        if dropna:
            self._df = self._df.dropna(subset=[img_col, time_col, event_col]).reset_index(drop=True)
            logger.info(f"Dropped rows with NaN. New dataset size: {len(self._df)}")


            self._paths = self._df[self.img_col].astype(str).tolist()


            self._cache_ds = data.PersistentDataset(
                data=[{"image": p} for p in self._paths],
                transform=loading_transforms(self.img_size, self.model_name),
                cache_dir=cache_dir,
            )

        self.time_col = time_col
        self.event_col = event_col


        self.time_arr = self._df[time_col].astype(np.float32)
        self.event_arr = self._df[event_col].astype(np.float32)


        self.normalize_time = normalize_time
        if normalize_time:
            if time_mean is not None and time_std is not None:
                self.time_mean = time_mean
                self.time_std = time_std
            else:
                self.time_mean = self.time_arr.mean()
                self.time_std = self.time_arr.std()
            self.time_arr = (self.time_arr - self.time_mean) / (self.time_std + 1e-7)
        else:
            self.time_mean = None
            self.time_std = None


        n_events = (self.event_arr == 1).sum()
        n_censored = (self.event_arr == 0).sum()
        logger.info(f"Survival dataset: {len(self._df)} samples, {n_events} events, {n_censored} censored")
        logger.info(f"Time range: [{self._df[time_col].min():.1f}, {self._df[time_col].max():.1f}]")

    def __getitem__(self, idx: int):
        try:
            sample = self._cache_ds[idx]

            if not self._valid_sample(sample, idx):

                return self._fallback_image(), torch.tensor(0.0), torch.tensor(0.0)


            sample = self.data_augmentation(sample)
            image = sample["image"] if isinstance(sample, dict) else sample


            time_val = self.time_arr.iloc[idx]
            event_val = self.event_arr.iloc[idx]

            time = torch.tensor(time_val, dtype=torch.float32)
            event = torch.tensor(event_val, dtype=torch.float32)

            return image, time, event

        except Exception as e:
            warnings.warn(f"[idx={idx}] Error during load/augment for file {self._paths[idx]}: {e}")
            return self._fallback_image(), torch.tensor(0.0), torch.tensor(0.0)

    def get_time_stats(self) -> Tuple[float, float]:

        if self.normalize_time:
            return self.time_mean, self.time_std
        else:
            return self._df[self.time_col].mean(), self._df[self.time_col].std()


def get_pretrain_dataloaders(cfg: Any, augs: Any) -> data.ThreadDataLoader:











    batch_size = cfg.data.batch_size
    cache_dir = cfg.data.cache_dir if cfg.data.cache_dir else None
    csv_file = cfg.data.train_csv_path

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    img_size = cfg.model.img_size
    patch_size = cfg.model.patch_size
    model_name = cfg.model.model_name
    in_chans = cfg.model.in_chans
    foreground_aware = cfg.model.foreground_aware if hasattr(cfg.model, "foreground_aware") else False






    mask_collator = MaskCollator(
        cfgs_mask=cfg.mask,
        crop_size=img_size,
        patch_size=patch_size,
        foreground_aware=foreground_aware,


    )


    train_ds = PretrainDataset(
        img_size=(list)(img_size),
        model_name=(str)(model_name),
        in_chans=(int)(in_chans),
        csv_file=csv_file,
        data_augmentation=augs,
        cache_dir=cache_dir,
    )
    sampler_train = data.DistributedSampler(
        train_ds,
        shuffle=True,
        num_replicas=num_tasks,
        rank=global_rank,
    )

    train_loader = data.ThreadDataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        collate_fn=mask_collator,

        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=True,
        worker_init_fn=mask_collator.worker_init_fn,
        persistent_workers=cfg.data.num_workers > 0,
    )

    return train_loader, mask_collator


def get_monitor_dataset(cfg: Any, augs: Any):

    imtrans, imvals, imtests = augs[0], augs[1], augs[2]


    batch_size = cfg.data.batch_size
    cache_dir = cfg.data.cache_dir if cfg.data.cache_dir else None


    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()


    df_train = pd.read_csv(cfg.data.train_csv_path)





    label_cols = cfg.data.label_cols





    eps = 1e-6
    label_arr = df_train[label_cols].to_numpy()
    class_counts = label_arr.sum(axis=0)



    N = len(label_arr)
    alpha = 1.0
    class_weights = (N / (class_counts + eps)) ** alpha

    sample_weights = (label_arr * class_weights).sum(axis=1) / (label_arr.sum(axis=1) + 1e-6)

    is_all_zero = (label_arr.sum(axis=1) == 0)

    neg_base = 0.1 * np.median(sample_weights[sample_weights > 0]) if (sample_weights > 0).any() else 1.0
    sample_weights = sample_weights.copy()
    sample_weights[is_all_zero] = neg_base

    sample_weights = sample_weights / (sample_weights.mean() + eps)


    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    logger.info(f"Sample weights: {sample_weights}")


    img_size = cfg.model.img_size
    patch_size = cfg.model.patch_size
    model_name = cfg.model.model_name
    in_chans = cfg.model.in_chans


    sample_size = cfg.data.samples_per_epoch
    sample_size = None if sample_size == -1 else sample_size // num_tasks

    train_ds = MonitorDataset(
        img_size=(list)(img_size),
        model_name=(str)(model_name),
        in_chans=(int)(in_chans),
        csv_file=cfg.data.train_csv_path,
        data_augmentation=imtrans,
        img_col=cfg.data.img_col,
        img_flag=cfg.data.img_flag,
        label_cols=label_cols,
        cache_dir=cache_dir,
    )






    sampler_train = data.DistributedWeightedRandomSampler(
        dataset=train_ds,
        weights=sample_weights,
        num_samples_per_rank=sample_size,
        rank=global_rank,
    )
    train_loader = data.ThreadDataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )


    val_ds = MonitorDataset(
        img_size=(list)(img_size),
        model_name=(str)(model_name),
        in_chans=(int)(in_chans),
        csv_file=cfg.data.val_csv_path,
        data_augmentation=imvals,
        img_col=cfg.data.img_col,
        img_flag=cfg.data.img_flag,
        label_cols=label_cols,
        cache_dir=cache_dir,
    )
    sampler_val = data.DistributedSampler(
        dataset=val_ds,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    val_loader = data.ThreadDataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        sampler=sampler_val,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )


    test_ds = MonitorDataset(
        img_size=(list)(img_size),
        model_name=(str)(model_name),
        in_chans=(int)(in_chans),
        csv_file=cfg.data.test_csv_path,
        data_augmentation=imtests,
        img_col=cfg.data.img_col,
        img_flag=cfg.data.img_flag,
        label_cols=label_cols,
        cache_dir=cache_dir,
    )
    sampler_test = data.DistributedSampler(
        dataset=test_ds,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    test_loader = data.ThreadDataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        sampler=sampler_test,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )

    return train_loader, val_loader, test_loader, test_ds._df.reset_index(drop=True)


def get_finetune_dataset(cfg: Any, augs: Any):





    imtrans, imvals, imtests = augs[0], augs[1], augs[2]


    batch_size = cfg.data.batch_size
    cache_dir = cfg.data.cache_dir if cfg.data.cache_dir else None


    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()


    df_train = pd.read_csv(cfg.data.train_csv_path)
    df_val = pd.read_csv(cfg.data.val_csv_path)
    df_test = pd.read_csv(cfg.data.test_csv_path)

    img_col = cfg.data.img_col
    label_col = cfg.data.label_col
    num_classes = cfg.data.num_classes


    df_train = clean_dataset(df_train, img_col, label_col)
    df_val = clean_dataset(df_val, img_col, label_col)
    df_test = clean_dataset(df_test, img_col, label_col)


    k_shot = cfg.data.get('k_shot', None)
    if k_shot is not None and k_shot > 0:
        few_shot_seed = cfg.meta.get('few_shot_seed', 42)
        df_train = few_shot_sample(
            df=df_train,
            label_col=label_col,
            k_shot=int(k_shot),
            num_classes=num_classes,
            seed=few_shot_seed,
        )


    labels_series = df_train[label_col].copy().fillna(0)
    label_arr = labels_series.astype(np.int64)


    if num_classes > 1:
        eps = 1e-6
        class_counts = np.bincount(label_arr, minlength=num_classes)
        N = len(label_arr)
        class_weights = N / (class_counts + eps)


        sample_weights = class_weights[label_arr]


        sample_weights = sample_weights / (sample_weights.mean() + eps)
    else:
        sample_weights = np.ones(len(label_arr), dtype=np.float32)
        class_counts = None
        class_weights = None

    logger.info(f"Finetune Class counts: {class_counts}")
    logger.info(f"Finetune Class weights: {class_weights}")


    img_size = cfg.model.img_size
    model_name = cfg.model.model_name
    in_chans = cfg.model.in_chans



    if num_classes == 1:
        mean = labels_series.mean()
        std = labels_series.std()
        logger.info(f"Regression label mean: {mean}, std: {std}")
    else:
        mean, std = None, None

    train_ds = FinetuneDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.train_csv_path,
        dataframe=df_train,
        data_augmentation=imtrans,
        img_col=img_col,
        label_col=label_col,
        num_classes=num_classes,
        cache_dir=cache_dir,
        dropna=False,


    )

    samples_per_epoch = cfg.data.get('samples_per_epoch')
    samples_per_epoch = None if samples_per_epoch == -1 else samples_per_epoch // num_tasks
    sampler_train = data.DistributedWeightedRandomSampler(
        dataset=train_ds,
        weights=sample_weights,
        num_samples_per_rank=samples_per_epoch,
        rank=global_rank,
    )
    train_loader = data.ThreadDataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )


    val_ds = FinetuneDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.val_csv_path,
        dataframe=df_val,
        data_augmentation=imvals,
        img_col=img_col,
        label_col=label_col,
        num_classes=num_classes,
        cache_dir=cache_dir,
        dropna=True,
    )
    sampler_val = data.DistributedSampler(
        dataset=val_ds, shuffle=False, num_replicas=num_tasks, rank=global_rank
    )
    val_loader = data.ThreadDataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        sampler=sampler_val,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )


    test_ds = FinetuneDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.test_csv_path,
        dataframe=df_test,
        data_augmentation=imtests,
        img_col=img_col,
        label_col=label_col,
        num_classes=num_classes,
        cache_dir=cache_dir,
        dropna=True,
    )
    sampler_test = data.DistributedSampler(
        dataset=test_ds, shuffle=False, num_replicas=num_tasks, rank=global_rank
    )
    test_loader = data.ThreadDataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        sampler=sampler_test,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )

    return train_loader, val_loader, test_loader


def few_shot_sample(
    df: pd.DataFrame,
    label_col: str,
    k_shot: int,
    num_classes: int,
    seed: int = 42,
) -> pd.DataFrame:



























    rng = np.random.default_rng(seed)

    if num_classes == 1:



        n_bins = 4
        values = df[label_col].astype(float)
        bin_edges = np.nanpercentile(values, np.linspace(0, 100, n_bins + 1))

        bin_edges[-1] += 1e-9
        parts = []
        for i in range(n_bins):
            mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
            bin_df = df[mask]
            n = min(k_shot, len(bin_df))
            if n == 0:
                warnings.warn(f"[few_shot] Regression bin {i} has 0 samples – skipped.")
                continue
            if n < k_shot:
                warnings.warn(
                    f"[few_shot] Regression bin {i} has only {n} samples (requested {k_shot})."
                )
            idx = rng.choice(len(bin_df), size=n, replace=False)
            parts.append(bin_df.iloc[idx])
        subset = pd.concat(parts, ignore_index=True)
        logger.info(
            f"[few_shot] Regression: sampled {len(subset)} rows "
            f"({k_shot} per quartile bin × {n_bins} bins, seed={seed})."
        )
    else:

        parts = []
        for cls in range(num_classes):
            cls_df = df[df[label_col] == cls]
            n = min(k_shot, len(cls_df))
            if n == 0:
                warnings.warn(f"[few_shot] Class {cls} has 0 samples – skipped.")
                continue
            if n < k_shot:
                warnings.warn(
                    f"[few_shot] Class {cls} has only {n} samples (requested {k_shot})."
                )
            idx = rng.choice(len(cls_df), size=n, replace=False)
            parts.append(cls_df.iloc[idx])
        subset = pd.concat(parts, ignore_index=True)
        logger.info(
            f"[few_shot] Classification: sampled {len(subset)} rows "
            f"({k_shot} per class × {num_classes} classes, seed={seed})."
        )

    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    return subset


def clean_dataset(df, img_col, label_col):
    df = df.dropna(subset=[img_col, label_col]).reset_index(drop=True)


    if 't1w' in img_col and 't1w_qc' in df.columns:
        df = df.dropna(subset=['t1w_qc'])
        df = df[df['t1w_qc'] == False]
    elif 't2w' in img_col and 't2w_qc' in df.columns:
        df = df.dropna(subset=['t2w_qc'])
        df = df[df['t2w_qc'] == False]
    elif 'flair' in img_col and 'flair_qc' in df.columns:
        df = df.dropna(subset=['flair_qc'])
        df = df[df['flair_qc'] == False]

    print(f"Clean dataset. New dataset size: {len(df)}")

    return df.reset_index(drop=True)


def clean_survival_dataset(df, img_col, time_col, event_col):

    df = df.dropna(subset=[img_col, time_col, event_col]).reset_index(drop=True)


    df = df[df[time_col] > 0].reset_index(drop=True)


    if 't1w' in img_col and 't1w_qc' in df.columns:
        df = df.dropna(subset=['t1w_qc'])
        df = df[df['t1w_qc'] == False]
    elif 't2w' in img_col and 't2w_qc' in df.columns:
        df = df.dropna(subset=['t2w_qc'])
        df = df[df['t2w_qc'] == False]
    elif 'flair' in img_col and 'flair_qc' in df.columns:
        df = df.dropna(subset=['flair_qc'])
        df = df[df['flair_qc'] == False]

    print(f"Clean survival dataset. New dataset size: {len(df)}")

    return df.reset_index(drop=True)


def get_survival_dataset(cfg: Any, augs: Any):











    imtrans, imvals, imtests = augs['train'], augs['val'], augs['test']


    batch_size = cfg.data.batch_size
    cache_dir = cfg.data.cache_dir if cfg.data.cache_dir else None


    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()


    img_col = cfg.data.img_col
    time_col = cfg.data.time_col
    event_col = cfg.data.event_col
    normalize_time = cfg.data.get('normalize_time', False)


    df_train = pd.read_csv(cfg.data.train_csv_path)
    df_val = pd.read_csv(cfg.data.val_csv_path)
    df_test = pd.read_csv(cfg.data.test_csv_path)

    df_train = clean_survival_dataset(df_train, img_col, time_col, event_col)
    df_val = clean_survival_dataset(df_val, img_col, time_col, event_col)
    df_test = clean_survival_dataset(df_test, img_col, time_col, event_col)


    img_size = cfg.model.img_size
    model_name = cfg.model.model_name
    in_chans = cfg.model.in_chans


    time_mean = df_train[time_col].mean() if normalize_time else None
    time_std = df_train[time_col].std() if normalize_time else None


    logger.info(f"Training set: {len(df_train)} samples, "
                f"{(df_train[event_col] == 1).sum()} events, "
                f"{(df_train[event_col] == 0).sum()} censored")
    logger.info(f"Validation set: {len(df_val)} samples")
    logger.info(f"Test set: {len(df_test)} samples")


    train_ds = SurvivalDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.train_csv_path,
        dataframe=df_train,
        data_augmentation=imtrans,
        img_col=img_col,
        time_col=time_col,
        event_col=event_col,
        cache_dir=cache_dir,
        dropna=False,
        normalize_time=normalize_time,
        time_mean=time_mean,
        time_std=time_std,
    )


    sampler_train = data.DistributedSampler(
        dataset=train_ds,
        shuffle=True,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    train_loader = data.ThreadDataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )


    val_ds = SurvivalDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.val_csv_path,
        dataframe=df_val,
        data_augmentation=imvals,
        img_col=img_col,
        time_col=time_col,
        event_col=event_col,
        cache_dir=cache_dir,
        dropna=False,
        normalize_time=normalize_time,
        time_mean=time_mean,
        time_std=time_std,
    )
    sampler_val = data.DistributedSampler(
        dataset=val_ds, shuffle=False, num_replicas=num_tasks, rank=global_rank
    )
    val_loader = data.ThreadDataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        sampler=sampler_val,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )


    test_ds = SurvivalDataset(
        img_size=list(img_size),
        model_name=str(model_name),
        in_chans=int(in_chans),
        csv_file=cfg.data.test_csv_path,
        dataframe=df_test,
        data_augmentation=imtests,
        img_col=img_col,
        time_col=time_col,
        event_col=event_col,
        cache_dir=cache_dir,
        dropna=False,
        normalize_time=normalize_time,
        time_mean=time_mean,
        time_std=time_std,
    )
    sampler_test = data.DistributedSampler(
        dataset=test_ds, shuffle=False, num_replicas=num_tasks, rank=global_rank
    )
    test_loader = data.ThreadDataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        sampler=sampler_test,
        collate_fn=custom_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
    )

    return train_loader, val_loader, test_loader
