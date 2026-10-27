from functools import partial

from networks import VisionTransformer

def vit_base_patch16_224(**kwargs):


    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model