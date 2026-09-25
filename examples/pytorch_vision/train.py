import argparse, os, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,10)
        )
    def forward(self,x): return self.net(x)

p=argparse.ArgumentParser()
p.add_argument("--epochs",type=int,default=1)
p.add_argument("--batch-size",type=int,default=64)
a=p.parse_args()

data_dir=os.getenv("SM_CHANNEL_TRAINING") or os.getenv("LOCAL_DATA_DIR") or "/tmp/cifar10"
os.makedirs(data_dir,exist_ok=True)
ds=datasets.CIFAR10(root=data_dir,train=True,download=True,transform=transforms.ToTensor())
ds=Subset(ds,range(min(2500,len(ds))))
dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=2)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
m=SmallCNN().to(device)
opt=torch.optim.Adam(m.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()
start=time.time(); seen=0
for _ in range(a.epochs):
    for x,y in dl:
        x,y=x.to(device),y.to(device)
        opt.zero_grad()
        loss=loss_fn(m(x),y)
        loss.backward()
        opt.step()
        seen+=len(y)
elapsed=time.time()-start
print("METRICS",json.dumps({
    "framework":"pytorch",
    "dataset":"CIFAR-10",
    "device":str(device),
    "elapsed_seconds":elapsed,
    "samples_per_second":seen/elapsed
}))
model_dir=os.getenv("SM_MODEL_DIR") or os.getenv("LOCAL_MODEL_DIR") or "/tmp/model"
os.makedirs(model_dir,exist_ok=True)
torch.save(m.state_dict(),os.path.join(model_dir,"model.pt"))
