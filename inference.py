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
    
    overall_results = []
    logic_results = []
    struct_results = []
    
    for item in item_list:
        eval_dataset = test_datasets[item]
        if args.dry_run:
            eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(min(2, len(eval_dataset)))))
        test_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Evaluate category
        res = evaluation_batch_loco(model, test_loader, device, max_ratio=0.01, resize_mask=256)
        
        overall_results.append({
            'item': item,
            'auroc_sp': res['auroc_sp'], 'ap_sp': res['ap_sp'], 'f1_sp': res['f1_sp'],
            'auroc_px': res['auroc_px'], 'ap_px': res['ap_px'], 'f1_px': res['f1_px'], 'aupro_px': res['aupro_px'],
            'combined': res['combined']
        })
        
        logic_results.append({
            'item': item,
            'auroc_sp': res['logic']['auroc_sp'], 'ap_sp': res['logic']['ap_sp'], 'f1_sp': res['logic']['f1_sp'],
            'auroc_px': res['logic']['auroc_px'], 'ap_px': res['logic']['ap_px'], 'f1_px': res['logic']['f1_px'], 'aupro_px': res['logic']['aupro_px']
        })
        
        struct_results.append({
            'item': item,
            'auroc_sp': res['struct']['auroc_sp'], 'ap_sp': res['struct']['ap_sp'], 'f1_sp': res['struct']['f1_sp'],
            'auroc_px': res['struct']['auroc_px'], 'ap_px': res['struct']['ap_px'], 'f1_px': res['struct']['f1_px'], 'aupro_px': res['struct']['aupro_px']
        })

    # Helper function to print a table
    def print_table(title, results_list, include_combined=False):
        print(f"\n==================== {title} ====================")
        header = f"{'Category':20s} | I-AUROC | I-AP    | I-F1    | P-AUROC | P-AP    | P-F1    | P-AUPRO"
        if include_combined:
            header += " | Combined"
        print(header)
        print("-" * len(header))
        
        auroc_sps, ap_sps, f1_sps = [], [], []
        auroc_pxs, ap_pxs, f1_pxs, aupro_pxs = [], [], [], []
        combineds = []
        
        for r in results_list:
            row = (
                f"{r['item']:20s} | {r['auroc_sp']:.4f}  | {r['ap_sp']:.4f}  | {r['f1_sp']:.4f}  | "
                f"{r['auroc_px']:.4f}  | {r['ap_px']:.4f}  | {r['f1_px']:.4f}  | {r['aupro_px']:.4f}"
            )
            if include_combined:
                row += f"  | {r['combined']:.4f}"
            print(row)
            
            auroc_sps.append(r['auroc_sp'])
            ap_sps.append(r['ap_sp'])
            f1_sps.append(r['f1_sp'])
            auroc_pxs.append(r['auroc_px'])
            ap_pxs.append(r['ap_px'])
            f1_pxs.append(r['f1_px'])
            aupro_pxs.append(r['aupro_px'])
            if include_combined:
                combineds.append(r['combined'])
                
        print("-" * len(header))
        mean_row = (
            f"{'MEAN':20s} | {np.mean(auroc_sps):.4f}  | {np.mean(ap_sps):.4f}  | {np.mean(f1_sps):.4f}  | "
            f"{np.mean(auroc_pxs):.4f}  | {np.mean(ap_pxs):.4f}  | {np.mean(f1_pxs):.4f}  | {np.mean(aupro_pxs):.4f}"
        )
        if include_combined:
            mean_row += f"  | {np.mean(combineds):.4f}"
        print(mean_row)
        print("=" * len(header) + "\n")

    print_table("Overall Performance (Mixed)", overall_results, include_combined=True)
    print_table("Logical Anomalies Performance", logic_results, include_combined=False)
    print_table("Structural Anomalies Performance", struct_results, include_combined=False)

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
