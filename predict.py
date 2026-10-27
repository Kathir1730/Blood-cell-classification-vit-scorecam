"""Predict the class of one cropped blood-cell image."""
import argparse, json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import timm
p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--class-names',default=None); args=p.parse_args()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
classes=json.loads(Path(args.class_names).read_text()) if args.class_names else ckpt['classes']; model=timm.create_model(ckpt.get('model_name','vit_base_patch16_224'),pretrained=False,num_classes=len(classes)); model.load_state_dict(ckpt['model']); model.to(device).eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
image=Image.open(args.image).convert('RGB'); x=tf(image).unsqueeze(0).to(device)
with torch.no_grad(): probs=model(x).softmax(1)[0]; idx=int(probs.argmax())
print(f'Prediction: {classes[idx]} | confidence: {probs[idx].item():.4f}')
