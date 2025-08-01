import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel
from mixed_optimizer import MixedOptimizer


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--local_test', action='store_true')
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['82'], type=str, choices=[str(s) for s in range(81, 87)])
    
    # Mixed Optimizer params
    parser.add_argument('--use_mixed_optimizer', action='store_true', help='使用混合优化器（稀疏embedding用SparseAdam，其他用AdamW）')
    parser.add_argument('--sparse_lr', default=None, type=float, help='稀疏参数学习率（默认使用--lr的值）')
    parser.add_argument('--dense_lr', default=None, type=float, help='密集参数学习率（默认使用--lr的值）')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    args = get_args()
    if args.local_test:
        data_path = '/home/leyankun/workspace/Tencent/TencentGR_1k'
    else:
        data_path = os.environ.get('TRAIN_DATA_PATH')
    print('开始运行')
    
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(  
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    print('模型初始化完成')
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    # 创建优化器
    if args.use_mixed_optimizer:
        # 使用混合优化器：稀疏embedding用SparseAdam，其他用AdamW
        sparse_lr = args.sparse_lr if args.sparse_lr is not None else args.lr
        dense_lr = args.dense_lr if args.dense_lr is not None else args.lr
        
        optimizer = MixedOptimizer(
            model=model,
            sparse_lr=sparse_lr,
            dense_lr=dense_lr,
            weight_decay=0.01,  # AdamW的权重衰减
            adam_betas=(0.9, 0.98),  # 保持与原始设置一致
            use_bf16=False,
            use_sparse_optimizer=True
        )
        print(f"✅ 使用混合优化器 - 稀疏学习率: {sparse_lr}, 密集学习率: {dense_lr}")
    else:
        # 使用传统的单一优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        print(f"📌 使用传统Adam优化器 - 学习率: {args.lr}")

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    
    # 时间预估相关变量
    train_start_time = time.time()
    total_steps = len(train_loader) * args.num_epochs
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # 计算时间预估
            current_time = time.time()
            elapsed_time = current_time - train_start_time
            avg_time_per_step = elapsed_time / (global_step + 1)
            remaining_steps = total_steps - (global_step + 1)
            estimated_remaining_time = remaining_steps * avg_time_per_step
            
            # 格式化时间 (转换为小时:分钟:秒格式)
            elapsed_hours, elapsed_remainder = divmod(int(elapsed_time), 3600)
            elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
            elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"
            
            eta_hours, eta_remainder = divmod(int(estimated_remaining_time), 3600)
            eta_minutes, eta_seconds = divmod(eta_remainder, 60)
            eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}"

            log_json = json.dumps(
                {
                    'global_step': global_step, 
                    'loss': loss.item(), 
                    'epoch': epoch, 
                    'time': current_time,
                    'elapsed_time': elapsed_str,
                    'eta': eta_str,
                    'progress_pct': round((global_step + 1) / total_steps * 100, 2),
                    'steps_per_sec': round(1/avg_time_per_step, 2)
                }
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            # for param in model.item_emb.parameters():
            #     loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

        # 更新hard negative池
        dataset.update_hard_negative_pool(model, args.device)

    print("Done")
    writer.close()
    log_file.close()
