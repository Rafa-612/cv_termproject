import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

from dataset import get_data_transforms, LOCODataset
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import evaluation_batch_loco

warnings.filterwarnings("ignore")

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Categories of MVTec LOCO AD
    item_list = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
    
    # Verify dataset exists
    for item in item_list:
        path = os.path.join(args.data_path, item)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset category folder '{path}' not found! "
                "Please run 'download_dataset.py' to download the dataset first."
            )
            
    # Setup image transformations
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    
    # Load test sets
    test_datasets = {}
    for item in item_list:
        item_path = os.path.join(args.data_path, item)
        test_data = LOCODataset(root=item_path, transform=data_transform, gt_transform=gt_transform, phase='test')
        test_datasets[item] = test_data
        print(f"Loaded category '{item}' with {len(test_data)} test samples.")
        
    # Load Backbone Encoder (DINOv2)
    encoder = vit_encoder.load(args.encoder_name)
    encoder = encoder.to(device)
    
    # Determine embedding dimensions based on encoder type
    if 'small' in args.encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("Invalid architecture name. Must be small, base, or large.")
        
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    if 'large' in args.encoder_name:
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    # Noisy MLP Bottleneck
    bottleneck = nn.ModuleList([
        bMlp(embed_dim, embed_dim * 4, embed_dim, drop=args.dropout)
    ])
    
    # Transformer Decoder (8 layers of blocks with LinearAttention2)
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim, 
            num_heads=num_heads, 
            mlp_ratio=4.0,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2
        ) for _ in range(8)
    ])
    
    # ViTill framework
    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=0
    )
    model = model.to(device)
    
    # Load model checkpoint
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{args.checkpoint_path}' not found!")
    
    print(f"Loading checkpoint weights from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")
    
    print("\nStarting evaluation...")
    
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
    auroc_logic_list, auroc_struct_list, auroc_both_list = [], [], []
    
    for item in item_list:
        eval_dataset = test_datasets[item]
        if args.dry_run:
            eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(min(2, len(eval_dataset)))))
        test_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Evaluate category
        results = evaluation_batch_loco(model, test_loader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, auroc_logic, auroc_struct, auroc_both = results
        
        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)
        auroc_logic_list.append(auroc_logic)
        auroc_struct_list.append(auroc_struct)
        auroc_both_list.append(auroc_both)
        
        print(
            f"{item:20s} | I-AUROC: {auroc_sp:.4f} | P-AUROC: {auroc_px:.4f} | P-AUPRO: {aupro_px:.4f} | "
            f"Logical: {auroc_logic:.4f} | Struct: {auroc_struct:.4f} | Combined: {auroc_both:.4f}"
        )
        
    print("-" * 120)
    print(
        f"{'MEAN':20s} | I-AUROC: {np.mean(auroc_sp_list):.4f} | P-AUROC: {np.mean(auroc_px_list):.4f} | P-AUPRO: {np.mean(aupro_px_list):.4f} | "
        f"Logical: {np.mean(auroc_logic_list):.4f} | Struct: {np.mean(auroc_struct_list):.4f} | Combined: {np.mean(auroc_both_list):.4f}"
    )
    print("-" * 120 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference and evaluation of Dinomaly on MVTec LOCO AD.')
    parser.add_argument('--data_path', type=str, default='data/mvtec_loco', help='Path to dataset directory.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint .pth file.')
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14', help='Backbone name.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--image_size', type=int, default=448, help='Image resizing resolution.')
    parser.add_argument('--crop_size', type=int, default=392, help='Crop size.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Noisy bottleneck dropout rate (0.2 default).')
    parser.add_argument('--dry_run', action='store_true', help='If set, only evaluate 2 samples per category for quick verification.')
    
    args = parser.parse_args()
    main(args)
