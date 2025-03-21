import copy
import glob
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms  

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from clip.loss import ClipLoss
from clip.model import CLIP
from clip_train.data import COCODataset
from clip_train.logger import setup_logging
from clip_train.params import parse_args
from clip_train.scheduler import const_lr, cosine_lr

def main(args):
    args = parse_args(args)

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + global_rank)

    model = CLIP(
        embed_dim=args.embed_dim,
        image_resolution=args.image_resolution,
        vision_layers=args.vision_layers,
        vision_width=args.vision_width,
        vision_patch_size=args.vision_patch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        transformer_width=args.transformer_width,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
    ).to(device, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    transform = transforms.Compose([
        transforms.Resize(args.image_resolution),
        transforms.CenterCrop(args.image_resolution),
        transforms.ToTensor(),
        transforms.Normalize(args.image_mean, args.image_std),
    ])
    dataset = COCODataset(args.data_dir, split="train", transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=sampler)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.wd,
    )
    
    total_steps = args.epochs * len(dataloader)
    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_steps, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, args.warmup_steps, total_steps)
    else:
        raise ValueError(f"Invalid learning rate scheduler: {args.lr_scheduler}")

    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(os.path.join(args.save_dir, "train.log"))
    
    if global_rank == 0:
        logger.info(f"Training with {world_size} GPUs")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Warmup steps: {args.warmup_steps}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Save frequency: {args.save_frequency}")

    global_step = 0
    model.zero_grad()
    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
            captions = captions.to(device, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)

            image_features, text_features = model(images, captions)
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.module.logit_scale.exp() if world_size > 1 else model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            labels = torch.arange(len(logits_per_image)).to(device)

            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss  = F.cross_entropy(logits_per_text, labels)

            loss = (image_loss + text_loss) / 2
            if world_size > 1:
                loss = loss.mean()
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_rank == 0:
                logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item()}")

                if batch_idx % args.save_frequency == 0:
                    model_path = os.path.join(args.save_dir, f"model_{epoch}_{global_step}.pth")
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, model_path)
                    logger.info(f"Saved model to {model_path}")

    if world_size > 1:
        dist.destroy_process_group()

    if global_rank == 0:
        logger.info("Training complete")


if __name__ == "__main__":
    main(sys.argv[1:])
