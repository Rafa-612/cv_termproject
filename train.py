import os
import argparse
import random
import warnings
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from functools import partial

from dataset import get_data_transforms, LOCODataset
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from optimizers import StableAdamW
from utils import global_cosine_hm_percent, evaluation_batch_loco, WarmCosineScheduler

warnings.filterwarnings("ignore")

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
        
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    setup_seed(1)
    
    # Categories of MVTec LOCO AD
    item_list = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
    
    # Verification check for data path
    for item in item_list:
        path = os.path.join(args.data_path, item)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset category folder '{path}' not found! "
                "Please run 'download_dataset.py' to download the dataset first."
            )
            
    # Setup logging
    save_path = os.path.join(args.save_dir, args.save_name)
    logger = get_logger(args.save_name, save_path)
    print_fn = logger.info
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_fn(f"Using device: {device}")
    
    # Transformations
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    
    # Load dataset
    print_fn("Loading MVTec LOCO datasets...")
    train_data_list = []
    test_data_list = []
    
    for i, item in enumerate(item_list):
        item_path = os.path.join(args.data_path, item)
        
        # Training set (uses normal images only)
        # Note: LOCODataset expects the category path (e.g. data/mvtec_loco/breakfast_box)
        train_data = LOCODataset(root=item_path, transform=data_transform, gt_transform=gt_transform, phase='train')
        train_data_list.append(train_data)
        
        # Testing set (contains logical/structural anomalies + ground truths)
        test_data = LOCODataset(root=item_path, transform=data_transform, gt_transform=gt_transform, phase='test')
        test_data_list.append(test_data)
        
    train_dataset = ConcatDataset(train_data_list)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        drop_last=True
    )
    
    print_fn(f"Total training images: {len(train_dataset)}")
    
    # Initialize Backbone Encoder (DINOv2)
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
    
    trainable_modules = nn.ModuleList([bottleneck, decoder])
    
    # Weight initialization for trainable modules
    for m in trainable_modules.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # Optimizer & Scheduler
    optimizer = StableAdamW(
        trainable_modules.parameters(),
        lr=args.lr, 
        betas=(0.9, 0.999), 
        weight_decay=1e-4, 
        amsgrad=True, 
        eps=1e-10
    )
    
    lr_scheduler = WarmCosineScheduler(
        optimizer, 
        base_value=args.lr, 
        final_value=args.lr / 10.0, 
        total_iters=args.total_iters,
        warmup_iters=100
    )
    
    print_fn("Starting unified training...")
    it = 0
    epoch = 0
    
    while it < args.total_iters:
        model.train()
        loss_list = []
        
        for img, _, _, _, _, _ in train_loader:
            img = img.to(device)
            
            en, de = model(img)
            
            # Linearly rise hard-mining ratio p from 0 to 0.9 in the first 1000 iterations
            p_final = 0.9
            p = min(p_final * it / 1000.0, p_final)
            
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_modules.parameters(), max_norm=0.1)
            
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
            
            it += 1
            
            # Print status periodically
            if it % 500 == 0:
                print_fn(f"Iteration {it}/{args.total_iters} | Loss: {loss.item():.4f}")
                
            # Perform intermediate evaluation and save model
            if it % args.eval_interval == 0 or it == args.total_iters:
                print_fn(f"\n--- Evaluation at Iteration {it} ---")
                
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
                auroc_logic_list, auroc_struct_list, auroc_both_list = [], [], []
                
                for item, test_data in zip(item_list, test_data_list):
                    if args.total_iters <= 10:
                        eval_dataset = torch.utils.data.Subset(test_data, list(range(min(2, len(test_data)))))
                    else:
                        eval_dataset = test_data
                    test_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                    
                    # Evaluate LOCO category
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
                    
                    print_fn(
                        f"{item:20s} | I-AUROC: {auroc_sp:.4f} | P-AUROC: {auroc_px:.4f} | P-AUPRO: {aupro_px:.4f} | "
                        f"Logical: {auroc_logic:.4f} | Struct: {auroc_struct:.4f} | Combined: {auroc_both:.4f}"
                    )
                    
                print_fn("-" * 120)
                print_fn(
                    f"{'MEAN':20s} | I-AUROC: {np.mean(auroc_sp_list):.4f} | P-AUROC: {np.mean(auroc_px_list):.4f} | P-AUPRO: {np.mean(aupro_px_list):.4f} | "
                    f"Logical: {np.mean(auroc_logic_list):.4f} | Struct: {np.mean(auroc_struct_list):.4f} | Combined: {np.mean(auroc_both_list):.4f}"
                )
                print_fn("-" * 120 + "\n")
                
                # Save checkpoint
                os.makedirs(save_path, exist_ok=True)
                ckpt_path = os.path.join(save_path, f"checkpoint_{it}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print_fn(f"Saved checkpoint to {ckpt_path}\n")
                
                model.train()
                
            if it == args.total_iters:
                break
                
        epoch += 1
        print_fn(f"Epoch {epoch} finished | Mean Loss: {np.mean(loss_list):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dinomaly on MVTec LOCO AD.')
    parser.add_argument('--data_path', type=str, default='data/mvtec_loco', help='Path to dataset directory.')
    parser.add_argument('--save_dir', type=str, default='saved_results', help='Directory to save output files.')
    parser.add_argument('--save_name', type=str, default='dinomaly_loco_base', help='Run name.')
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14', help='Backbone name.')
    parser.add_argument('--total_iters', type=int, default=10000, help='Total training iterations.')
    parser.add_argument('--eval_interval', type=int, default=5000, help='Evaluation interval in iterations.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--image_size', type=int, default=448, help='Image resizing resolution.')
    parser.add_argument('--crop_size', type=int, default=392, help='Crop size.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Base learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Noisy bottleneck dropout rate.')
    
    args = parser.parse_args()
    main(args)
