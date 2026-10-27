"""Score-CAM-style explanation using final-block ViT patch-token channels.

This adapts Score-CAM to ViT token maps: final transformer patch-token channels
are reshaped to a grid, upsampled to input resolution, and used as input masks.
This is computationally expensive (one masked forward pass per selected channel).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import timm
p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--class-names',default=None); p.add_argument('--output',default='scorecam.png'); p.add_argument('--max-channels',type=int,default=64); args=p.parse_args()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
classes=json.loads(Path(args.class_names).read_text()) if args.class_names else ckpt['classes']; model=timm.create_model(ckpt.get('model_name','vit_base_patch16_224'),pretrained=False,num_classes=len(classes)); model.load_state_dict(ckpt['model']); model.to(device).eval()
mean=torch.tensor([.5,.5,.5],device=device).view(1,3,1,1); std=torch.tensor([.5,.5,.5],device=device).view(1,3,1,1)
img=Image.open(args.image).convert('RGB').resize((224,224)); raw=transforms.ToTensor()(img).unsqueeze(0).to(device); x=(raw-mean)/std
acts=[]
def hook(_m,_inp,out): acts.append(out.detach())
handle=model.blocks[-1].register_forward_hook(hook)
with torch.no_grad(): base_logits=model(x); target=int(base_logits.argmax(1).item())
handle.remove(); tokens=acts[-1]
# Remove CLS token, then reshape patch tokens (14x14 for 224px, patch16).
patch=tokens[:,1:,:]
grid=int(patch.shape[1]**0.5)
if grid*grid != patch.shape[1]: raise RuntimeError(f'Patch token count {patch.shape[1]} is not square; cannot reshape.')
features=patch[0].transpose(0,1).reshape(1,-1,grid,grid); features=F.relu(features)
# Select channels with strongest spatial variation, then score each masked input.
variation=features.flatten(2).std(dim=2)[0]; k=min(args.max_channels,features.shape[1]); inds=torch.topk(variation,k).indices
maps=F.interpolate(features[:,inds],size=(224,224),mode='bilinear',align_corners=False)[0]
lo=maps.amin(dim=(1,2),keepdim=True); hi=maps.amax(dim=(1,2),keepdim=True); masks=(maps-lo)/(hi-lo+1e-8)
weighted=[]
with torch.no_grad():
 for start in range(0,k,8):
  batch_masks=masks[start:start+8].unsqueeze(1)
  masked=raw*batch_masks
  masked=(masked-mean)/std
  scores=model(masked)[:,target].softmax(dim=0)
  weighted.extend((scores[:,None,None]*masks[start:start+8]).unbind(0))
heat=torch.stack(weighted).sum(0); heat=(heat-heat.min())/(heat.max()-heat.min()+1e-8); heat=heat.cpu().numpy()
orig=np.asarray(img).astype(np.float32)/255.; cmap=plt.get_cmap('jet')(heat)[...,:3]; overlay=np.clip(.55*orig+.45*cmap,0,1)
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(1,3,figsize=(12,4)); ax[0].imshow(orig); ax[0].set_title('Input'); ax[1].imshow(heat,cmap='jet'); ax[1].set_title('ViT token Score-CAM'); ax[2].imshow(overlay); ax[2].set_title(f'Overlay — {classes[target]}');
for a in ax: a.axis('off')
fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig); print(f'Predicted class: {classes[target]}; saved explanation to {out}')
