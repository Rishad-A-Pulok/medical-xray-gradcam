"""Reproducible pediatric chest X-ray binary classification and Grad-CAM.
Usage: python -m src.pipeline prepare --data-root data/chest_xray
       python -m src.pipeline train --model resnet18 --epochs 8
       python -m src.pipeline evaluate --model resnet18
       python -m src.pipeline gradcam --image path/to/image.jpeg
"""
import argparse, json, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
LABELS=['NORMAL','PNEUMONIA']

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def prepare(data_root):
    base=Path(data_root).resolve()
    records=[]
    for original in ['train','val','test']:
        for label in LABELS:
            folder=base/original/label
            if not folder.is_dir(): raise FileNotFoundError(folder)
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in ['.jpg','.jpeg','.png']: continue
                try:
                    with Image.open(p) as im: im.verify()
                except Exception: continue
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                records.append({'path':str(p),'label':LABELS.index(label),'original_split':original,'sha256':digest})
    df=pd.DataFrame(records)
    # Exact duplicates crossing original test boundary make a valid held-out test impossible.
    trainhash=set(df.loc[df.original_split!='test','sha256'])
    testhash=set(df.loc[df.original_split=='test','sha256'])
    if trainhash & testhash: raise ValueError('Exact duplicates cross the original test boundary; investigate before continuing')
    # Preserve original test, combine original train+tiny val into development pool.
    development=df[df.original_split!='test'].drop_duplicates('sha256').copy()
    test=df[df.original_split=='test'].drop_duplicates('sha256').copy()
    train_idx,val_idx=train_test_split(np.arange(len(development)),test_size=.15,stratify=development.label,random_state=42)
    development.iloc[train_idx].assign(split='train').to_csv(ROOT/'data/train.csv',index=False)
    development.iloc[val_idx].assign(split='val').to_csv(ROOT/'data/val.csv',index=False)
    test.assign(split='test').to_csv(ROOT/'data/test.csv',index=False)
    print('Counts:',len(train_idx),len(val_idx),len(test))
    print('Warning: filenames are not reliable patient IDs; patient-level separation is not guaranteed.')

class XRays(Dataset):
    def __init__(self, split, augment=False):
        self.df=pd.read_csv(ROOT/f'data/{split}.csv')
        ops=[transforms.Resize((224,224))]
        if augment: ops += [transforms.RandomRotation(7),transforms.RandomAffine(degrees=0,translate=(.03,.03))]
        ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]
        self.tf=transforms.Compose(ops)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.path) as img: x=self.tf(img.convert('RGB'))
        return x,int(r.label)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.fc=nn.Linear(64,2)
    def forward(self,x): return self.fc(self.features(x).flatten(1))

def build(name,pretrained=False):
    if name=='cnn': return SmallCNN()
    if name=='resnet18':
        model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc=nn.Linear(model.fc.in_features,2)
        return model
    raise ValueError(name)

def loader(split,batch,augment=False): return DataLoader(XRays(split,augment),batch_size=batch,shuffle=augment,num_workers=2,pin_memory=torch.cuda.is_available())

def run_epoch(model,dl,lossfn,device,opt=None):
    model.train(opt is not None)
    total=correct=loss_total=0
    for x,y in dl:
        x=x.to(device);y=y.to(device)
        with torch.set_grad_enabled(opt is not None):
            logits=model(x);loss=lossfn(logits,y)
            if opt is not None:
                opt.zero_grad();loss.backward();opt.step()
        total+=len(y);correct+=(logits.argmax(1)==y).sum().item();loss_total+=loss.item()*len(y)
    return {'loss':loss_total/total,'accuracy':correct/total}

def train(args):
    seed_all();device='cuda' if torch.cuda.is_available() else 'cpu'
    tr=loader('train',args.batch,True);va=loader('val',args.batch)
    model=build(args.model,pretrained=args.model=='resnet18').to(device)
    labels=tr.dataset.df.label.to_numpy(); counts=np.bincount(labels,minlength=2)
    weights=torch.tensor(len(labels)/(2*np.maximum(counts,1)),dtype=torch.float32,device=device)
    lossfn=nn.CrossEntropyLoss(weight=weights)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float('inf');history=[];bad=0
    (ROOT/'models').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    for ep in range(args.epochs):
        a=run_epoch(model,tr,lossfn,device,opt);b=run_epoch(model,va,lossfn,device)
        history.append({'epoch':ep+1,'train_loss':a['loss'],'train_accuracy':a['accuracy'],'val_loss':b['loss'],'val_accuracy':b['accuracy']})
        print(history[-1])
        if b['loss']<best:
            best=b['loss'];bad=0
            torch.save({'model':model.state_dict(),'name':args.model,'classes':LABELS},ROOT/f'models/{args.model}.pt')
        else:
            bad+=1
            if bad>=4: break
    pd.DataFrame(history).to_csv(ROOT/f'results/{args.model}_history.csv',index=False)

def load_model(name):
    ck=torch.load(ROOT/f'models/{name}.pt',map_location='cpu',weights_only=True)
    model=build(name);model.load_state_dict(ck['model']);model.eval()
    return model

def evaluate(args):
    model=load_model(args.model);device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    ys=[];probs=[];paths=[]
    ds=XRays('test');dl=DataLoader(ds,batch_size=args.batch,shuffle=False,num_workers=2)
    with torch.no_grad():
        for x,y in dl:
            p=model(x.to(device)).softmax(1)[:,1].cpu().numpy()
            ys.extend(y.numpy());probs.extend(p)
    ys=np.array(ys);probs=np.array(probs);pred=(probs>=.5).astype(int)
    report=classification_report(ys,pred,target_names=LABELS,output_dict=True,zero_division=0)
    report['roc_auc']=float(roc_auc_score(ys,probs)) if len(set(ys))==2 else None
    report['average_precision']=float(average_precision_score(ys,probs)) if len(set(ys))==2 else None
    (ROOT/'results').mkdir(exist_ok=True)
    with open(ROOT/f'results/{args.model}_metrics.json','w') as f:json.dump(report,f,indent=2)
    pd.DataFrame({'path':ds.df.path,'true':ys,'prob_pneumonia':probs,'pred':pred}).to_csv(ROOT/f'results/{args.model}_predictions.csv',index=False)
    cm=confusion_matrix(ys,pred,labels=[0,1]);fig,ax=plt.subplots();ax.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
    ax.set_xticks([0,1],LABELS);ax.set_yticks([0,1],LABELS);ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(ROOT/f'results/{args.model}_confusion.png');plt.close(fig)
    if len(set(ys))==2:
        fpr,tpr,_=roc_curve(ys,probs);precision,recall,_=precision_recall_curve(ys,probs)
        for x,y,title,filename,xlabel,ylabel in [(fpr,tpr,'ROC', 'roc','False positive rate','True positive rate'),(recall,precision,'Precision–Recall','pr','Recall','Precision')]:
            fig,ax=plt.subplots();ax.plot(x,y);ax.set(title=title,xlabel=xlabel,ylabel=ylabel);fig.tight_layout();fig.savefig(ROOT/f'results/{args.model}_{filename}.png');plt.close(fig)
    print(json.dumps(report,indent=2))

def gradcam(args):
    if args.model!='resnet18':raise ValueError('Grad-CAM example targets resnet18 layer4[-1]')
    model=load_model(args.model);model.eval();activations=[];gradients=[]
    layer=model.layer4[-1]
    h1=layer.register_forward_hook(lambda module,inputs,output:activations.append(output))
    h2=layer.register_full_backward_hook(lambda module,gin,gout:gradients.append(gout[0]))
    with Image.open(args.image) as im: img=im.convert('RGB').resize((224,224))
    x=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])(img).unsqueeze(0)
    logits=model(x);target=int(logits.argmax(1).item()) if args.target is None else args.target
    model.zero_grad();logits[0,target].backward()
    act=activations[0].detach()[0];grad=gradients[0].detach()[0]
    cam=torch.relu((act*grad.mean(dim=(1,2),keepdim=True)).sum(dim=0)).numpy()
    cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    cam=np.asarray(Image.fromarray(np.uint8(cam*255)).resize((224,224),Image.Resampling.BILINEAR))/255
    h1.remove();h2.remove()
    fig,ax=plt.subplots(1,2,figsize=(9,4));ax[0].imshow(img);ax[0].set_title('Input X-ray');ax[1].imshow(img);ax[1].imshow(cam,cmap='jet',alpha=.4,vmin=0,vmax=1);ax[1].set_title(f'Grad-CAM target: {LABELS[target]}')
    for a in ax:a.axis('off')
    (ROOT/'results').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(ROOT/'results/gradcam_example.png',dpi=180);plt.close(fig)
    print('Saved results/gradcam_example.png; attribution is not lesion localization.')

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--data-root',required=True)
    for cmd in ['train','evaluate','gradcam']:
        a=sub.add_parser(cmd);a.add_argument('--model',choices=['cnn','resnet18'],default='resnet18');a.add_argument('--batch',type=int,default=32)
        if cmd=='train':a.add_argument('--epochs',type=int,default=8);a.add_argument('--lr',type=float,default=1e-4)
        if cmd=='gradcam':a.add_argument('--image',required=True);a.add_argument('--target',type=int,choices=[0,1],default=None)
    args=p.parse_args();{'prepare':lambda:prepare(args.data_root),'train':lambda:train(args),'evaluate':lambda:evaluate(args),'gradcam':lambda:gradcam(args)}[args.command]()
