"""Train a ViT-Base/Patch16 model from scratch on ImageFolder data."""
import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data'); p.add_argument('--output-dir', default='outputs')
    p.add_argument('--epochs', type=int, default=50); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42); args = p.parse_args()
    seed_everything(args.seed); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_tf = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(brightness=.12, contrast=.12), transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)])
    eval_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)])
    train_ds = datasets.ImageFolder(Path(args.data_dir)/'train', transform=train_tf)
    val_ds = datasets.ImageFolder(Path(args.data_dir)/'val', transform=eval_tf)
    if train_ds.classes != val_ds.classes: raise ValueError(f"Class mismatch: train={train_ds.classes}, val={val_ds.classes}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type=='cuda')
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type=='cuda')
    # pretrained=False is intentional: train from scratch.
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = 0.0
    (out/'classes.json').write_text(json.dumps(train_ds.classes, indent=2))
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=0.; correct=total=0
        for x,y in train_dl:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
            loss_sum += loss.item()*y.size(0); correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
        model.eval(); vc=vt=0; vloss=0.
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device); logits=model(x); vloss += criterion(logits,y).item()*y.size(0); vc += (logits.argmax(1)==y).sum().item(); vt += y.size(0)
        scheduler.step(); train_acc=correct/max(total,1); val_acc=vc/max(vt,1)
        print(f'Epoch {epoch:03d}/{args.epochs} train_loss={loss_sum/max(total,1):.4f} train_acc={train_acc:.4f} val_loss={vloss/max(vt,1):.4f} val_acc={val_acc:.4f}')
        if val_acc > best:
            best=val_acc; torch.save({'model':model.state_dict(),'classes':train_ds.classes,'model_name':'vit_base_patch16_224'},out/'best_model.pth')
    print(f'Best validation accuracy: {best:.4f}. Checkpoint: {out / "best_model.pth"}')

if __name__ == '__main__': main()
