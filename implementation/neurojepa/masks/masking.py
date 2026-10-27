import math
from logging import getLogger
from multiprocessing import Value

import torch
import torch.nn.functional as F

_GLOBAL_SEED = 0
logger = getLogger()


def compute_foreground_patches(
    volumes,
    patch_size,
    threshold=0.0,
    min_foreground_fraction=0.1,
):




    pH, pW, pD = patch_size
    B = volumes.shape[0]
    H, W, D = volumes.shape[2], volumes.shape[3], volumes.shape[4]

    assert H % pH == 0 and W % pW == 0 and D % pD == 0, (
        f"Volume spatial dims ({H}, {W}, {D}) must be evenly divisible by "
        f"patch_size ({pH}, {pW}, {pD})."
    )


    vol = volumes.abs().amax(dim=1)


    vol_flat = vol.reshape(B, -1)
    subsample_stride = max(1, vol_flat.shape[1] // 100000)
    vol_sub = vol_flat[:, ::subsample_stride].float()

    p2  = torch.quantile(vol_sub, 0.02, dim=1)
    p98 = torch.quantile(vol_sub, 0.98, dim=1)
    dyn_thresh = 0.1 * (p98 - p2) + p2

    dyn_thresh = torch.where(p98 > p2, dyn_thresh, torch.full_like(dyn_thresh, threshold))
    dyn_thresh = dyn_thresh.view(B, 1, 1, 1)


    voxel_mask = (vol > dyn_thresh).float().unsqueeze(1)


    fg_frac = F.avg_pool3d(
        voxel_mask,
        kernel_size=(pH, pW, pD),
        stride=(pH, pW, pD)
    ).squeeze(1)

    return fg_frac >= min_foreground_fraction

class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224, 224),
        patch_size=(16, 16, 16),
        foreground_aware=False,
        foreground_threshold=0.0,
        min_foreground_fraction=0.1,
    ):
        super(MaskCollator, self).__init__()
        self.patch_size = patch_size
        self.foreground_aware = foreground_aware
        self.foreground_threshold = foreground_threshold
        self.min_foreground_fraction = min_foreground_fraction

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                patch_size=patch_size,
                spatial_pred_mask_scale=m.get("spatial_scale"),
                depth_pred_mask_scale=m.get("depth_scale"),
                aspect_ratio=m.get("aspect_ratio"),
                npred=m.get("num_blocks"),
                total_mask_ratio=m.get("total_mask_ratio"),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def worker_init_fn(self, worker_id: int):

        for mask_generator in self.mask_generators:
            mask_generator.set_worker_seed_offset(worker_id)

    def __call__(self, batch):
        batch_size = len(batch)
        if batch_size == 0:
            raise ValueError("Detect batch size of 0 in MaskCollator")

        collated_batch = torch.utils.data.default_collate(batch)


        fg_mask = None
        if self.foreground_aware:
            volumes = collated_batch[0] if isinstance(collated_batch, (list, tuple)) else collated_batch
            if isinstance(volumes, torch.Tensor) and volumes.ndim == 5:
                fg_mask = compute_foreground_patches(
                    volumes,
                    self.patch_size,
                    threshold=self.foreground_threshold,
                    min_foreground_fraction=self.min_foreground_fraction,
                )

        collated_masks_pred, collated_masks_enc = [], []
        for mask_generator in self.mask_generators:
            masks_enc, masks_pred = mask_generator(batch_size, foreground_mask=fg_mask)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)


        fg_flat = fg_mask.flatten(1).float() if fg_mask is not None else None

        return (collated_batch, collated_masks_enc, collated_masks_pred, fg_flat)


class _MaskGenerator(object):





    def __init__(
        self,
        crop_size=(224, 224, 224),
        patch_size=(16, 16, 16),
        spatial_pred_mask_scale=(0.15, 0.15),
        depth_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        total_mask_ratio=0.7,
        npred=1,
        min_context_ratio=0.15,
    ):
        super(_MaskGenerator, self).__init__()

        self.crop_size = crop_size
        self.patch_size = patch_size

        self.height = crop_size[0] // patch_size[0]
        self.width  = crop_size[1] // patch_size[1]
        self.depth  = crop_size[2] // patch_size[2]

        assert self.height > 0 and self.width > 0 and self.depth > 0,\
            f"Invalid patch grid dimensions: ({self.height}, {self.width}, {self.depth})"

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.depth_pred_mask_scale = depth_pred_mask_scale

        self.num_patches = self.height * self.width * self.depth
        self.total_mask_ratio = total_mask_ratio
        self.npred = npred



        self.target_pred = int(self.num_patches * total_mask_ratio)
        self.target_enc  = self.num_patches - self.target_pred


        self.min_context_ratio = min_context_ratio
        self.min_context_num = max(10, int(self.num_patches * min_context_ratio))
        assert self.target_enc >= self.min_context_num, (
            f"target_enc={self.target_enc} < min_context={self.min_context_num}. "
            f"Reduce total_mask_ratio ({total_mask_ratio}) or min_context_ratio ({min_context_ratio})."
        )


        self._min_depth_scale = max(
            depth_pred_mask_scale[0], 1.0 / max(1, self.depth)
        )
        self._min_spatial_scale = max(
            spatial_pred_mask_scale[0], 1.0 / max(1, self.height * self.width)
        )


        self._worker_seed_offset = 0
        self._itr_counter = Value("i", -1)

    def set_worker_seed_offset(self, worker_id):


        self._worker_seed_offset = worker_id * 10_000_000

    def step(self):

        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            v = self._itr_counter.value
        return v + self._worker_seed_offset

    def _sample_block_size(self, generator):








        _rand = torch.rand(1, generator=generator).item()
        max_d = self.depth_pred_mask_scale[1]
        depth_scale = self._min_depth_scale + _rand * (max_d - self._min_depth_scale)
        d = max(1, min(int(self.depth * depth_scale), self.depth))


        _rand = torch.rand(1, generator=generator).item()
        max_s = self.spatial_pred_mask_scale[1]
        spatial_scale = self._min_spatial_scale + _rand * (max_s - self._min_spatial_scale)
        spatial_area = max(1, int(self.height * self.width * spatial_scale))


        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = self.aspect_ratio
        if min_ar > 0 and max_ar > 0:
            log_ar = math.log(min_ar) + _rand * (math.log(max_ar) - math.log(min_ar))
            ar = math.exp(log_ar)
        else:
            ar = 1.0

        h = max(1, min(int(round(math.sqrt(spatial_area * ar))), self.height))
        w = max(1, min(int(round(math.sqrt(spatial_area / ar))), self.width))

        return d, h, w




    @staticmethod
    def _boundary_indices(mask_3d):




        H, W, D = mask_3d.shape

        padded = F.pad(mask_3d.unsqueeze(0).unsqueeze(0).float(), (1,1,1,1,1,1), value=1.0)

        neighbour_masked = (
            (padded[:, :, :-2, 1:-1, 1:-1] == 0) |
            (padded[:, :, 2:,  1:-1, 1:-1] == 0) |
            (padded[:, :, 1:-1, :-2, 1:-1] == 0) |
            (padded[:, :, 1:-1, 2:,  1:-1] == 0) |
            (padded[:, :, 1:-1, 1:-1, :-2] == 0) |
            (padded[:, :, 1:-1, 1:-1, 2:]  == 0)
        ).squeeze(0).squeeze(0)

        boundary = mask_3d & neighbour_masked
        return boundary.flatten().nonzero(as_tuple=True)[0]

    @staticmethod
    def _inner_boundary_indices(mask_3d):



        H, W, D = mask_3d.shape
        padded = F.pad(mask_3d.unsqueeze(0).unsqueeze(0).float(), (1,1,1,1,1,1), value=0.0)
        neighbour_unmasked = (
            (padded[:, :, :-2, 1:-1, 1:-1] == 1) |
            (padded[:, :, 2:,  1:-1, 1:-1] == 1) |
            (padded[:, :, 1:-1, :-2, 1:-1] == 1) |
            (padded[:, :, 1:-1, 2:,  1:-1] == 1) |
            (padded[:, :, 1:-1, 1:-1, :-2] == 1) |
            (padded[:, :, 1:-1, 1:-1, 2:]  == 1)
        ).squeeze(0).squeeze(0)

        inner = (~mask_3d) & neighbour_unmasked
        return inner.flatten().nonzero(as_tuple=True)[0]


    def __call__(self, batch_size, foreground_mask=None):











        seed = self.step()



        masks_enc_list, masks_pred_list = [], []

        for si in range(batch_size):

            g = torch.Generator()
            g.manual_seed(seed * 1_000_003 + si)

            mask = torch.ones(self.height, self.width, self.depth, dtype=torch.bool)

            for _ in range(self.npred):
                d, h, w = self._sample_block_size(g)

                top   = torch.randint(0, max(1, self.height - h + 1), (1,), generator=g).item()
                left  = torch.randint(0, max(1, self.width  - w + 1), (1,), generator=g).item()
                start = torch.randint(0, max(1, self.depth  - d + 1), (1,), generator=g).item()

                mask[top:top+h, left:left+w, start:start+d] = False




            n_keep = mask.sum().item()
            mask_flat = mask.flatten()


            fg_flat = (
                foreground_mask[si].flatten().bool()
                if foreground_mask is not None else None
            )

            if n_keep > self.target_enc:

                n_remove = n_keep - self.target_enc
                while n_remove > 0:
                    boundary = self._boundary_indices(mask)
                    if len(boundary) == 0:
                        break

                    if fg_flat is not None:
                        is_bg = ~fg_flat[boundary]
                        bg = boundary[is_bg]
                        fg = boundary[~is_bg]
                        bg = bg[torch.randperm(len(bg), generator=g)]
                        fg = fg[torch.randperm(len(fg), generator=g)]
                        boundary = torch.cat([bg, fg])
                    else:
                        boundary = boundary[torch.randperm(len(boundary), generator=g)]
                    take = min(n_remove, len(boundary))
                    mask_flat[boundary[:take]] = False
                    mask = mask_flat.view(self.height, self.width, self.depth)
                    n_remove -= take

            elif n_keep < self.target_enc:

                n_add = self.target_enc - n_keep
                while n_add > 0:
                    inner = self._inner_boundary_indices(mask)
                    if len(inner) == 0:
                        break

                    if fg_flat is not None:
                        is_fg = fg_flat[inner]
                        fg = inner[is_fg]
                        bg = inner[~is_fg]
                        fg = fg[torch.randperm(len(fg), generator=g)]
                        bg = bg[torch.randperm(len(bg), generator=g)]
                        inner = torch.cat([fg, bg])
                    else:
                        inner = inner[torch.randperm(len(inner), generator=g)]
                    take = min(n_add, len(inner))
                    mask_flat[inner[:take]] = True
                    mask = mask_flat.view(self.height, self.width, self.depth)
                    n_add -= take




            mask_flat = mask.flatten()
            n_keep = mask_flat.sum().item()

            if n_keep > self.target_enc:

                kept_idx = mask_flat.nonzero(as_tuple=True)[0]
                kept_idx = kept_idx[torch.randperm(len(kept_idx), generator=g)]
                n_remove = n_keep - self.target_enc
                mask_flat[kept_idx[:n_remove]] = False
            elif n_keep < self.target_enc:

                masked_idx = (~mask_flat).nonzero(as_tuple=True)[0]
                masked_idx = masked_idx[torch.randperm(len(masked_idx), generator=g)]
                n_add = self.target_enc - n_keep
                mask_flat[masked_idx[:n_add]] = True


            enc_idx  = mask_flat.nonzero(as_tuple=True)[0]
            pred_idx = (~mask_flat).nonzero(as_tuple=True)[0]


            enc_idx  = enc_idx[torch.randperm(len(enc_idx), generator=g)]
            pred_idx = pred_idx[torch.randperm(len(pred_idx), generator=g)]

            masks_enc_list.append(enc_idx)
            masks_pred_list.append(pred_idx)


        masks_enc  = torch.stack(masks_enc_list)
        masks_pred = torch.stack(masks_pred_list)

        return masks_enc, masks_pred
