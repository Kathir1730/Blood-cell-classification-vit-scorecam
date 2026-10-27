"""Evaluate a saved ViT checkpoint on a labelled ImageFolder."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import timm

p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--class-names',default=None); p.add_argument('--output-dir',default='outputs/evaluation'); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--workers',type=int,default=2); args=p.parse_args()
out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
ds=datasets.ImageFolder(args.data_dir,transform=tf); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False); classes=json.loads(Path(args.class_names).read_text()) if args.class_names else ckpt.get('classes',ds.classes)
if classes != ds.classes: print('Warning: checkpoint class order differs from test folder order; mapping labels by folder names.')
model=timm.create_model(ckpt.get('model_name','vit_base_patch16_224'),pretrained=False,num_classes=len(classes)); model.load_state_dict(ckpt['model']); model.to(device).eval(); ytrue=[]; ypred=[]; rows=[]
with torch.no_grad():
 for x,y in dl:
  logits=model(x.to(device)); pred=logits.argmax(1).cpu().tolist()
  ytrue.extend(y.tolist()); ypred.extend(pred)
  rows.extend({'path':ds.samples[len(rows)+i][0],'true_label':ds.classes[int(y[i])],'predicted_label':classes[pred[i]]} for i in range(len(pred)))
# ImageFolder label indices follow alphabetically sorted folder names.
labels=list(range(len(ds.classes))); report=classification_report(ytrue,ypred,labels=labels,target_names=ds.classes,zero_division=0)
(out/'classification_report.txt').write_text(f'Accuracy: {accuracy_score(ytrue,ypred):.6f}\n\n{report}')
pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False); cm=confusion_matrix(ytrue,ypred,labels=labels)
fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax); ax.set(xticks=labels,yticks=labels,xticklabels=ds.classes,yticklabels=ds.classes,xlabel='Predicted',ylabel='True',title='Confusion Matrix'); plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=180); plt.close(fig); print(f'Accuracy: {accuracy_score(ytrue,ypred):.4f}\n{report}\nSaved outputs to {out}')
