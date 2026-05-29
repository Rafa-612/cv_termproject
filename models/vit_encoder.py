import torch
import timm

def load(name):
    """
    Loads pretrained backbones from timm.
    Handles DINOv2 models with registers and dynamic input resolutions.
    Adds a `prepare_tokens` method and `num_register_tokens` attribute for compatibility.
    """
    print(f"Loading backbone '{name}' via timm...")
    if name == 'dinov2reg_vit_base_14':
        timm_name = 'vit_base_patch14_reg4_dinov2.lvd142m'
    elif name == 'dinov2reg_vit_small_14':
        timm_name = 'vit_small_patch14_reg4_dinov2.lvd142m'
    elif name == 'dinov2reg_vit_large_14':
        timm_name = 'vit_large_patch14_reg4_dinov2.lvd142m'
    elif name == 'dinov2_vit_base_14':
        timm_name = 'vit_base_patch14_dinov2.lvd142m'
    elif name == 'dinov2_vit_small_14':
        timm_name = 'vit_small_patch14_dinov2.lvd142m'
    elif name == 'dinov2_vit_large_14':
        timm_name = 'vit_large_patch14_dinov2.lvd142m'
    else:
        # Fallback to standard timm names if a custom name is passed
        timm_name = name
        
    model = timm.create_model(timm_name, pretrained=True, dynamic_img_size=True)
    
    # Configure registers and prefix tokens
    if 'reg4' in timm_name:
        model.num_register_tokens = 4
    else:
        model.num_register_tokens = 0
        
    # Bind a DINOv2-compatible prepare_tokens method to the model
    def prepare_tokens(x):
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = model.norm_pre(x)
        return x
        
    model.prepare_tokens = prepare_tokens
    
    print("Backbone loaded successfully.")
    return model
