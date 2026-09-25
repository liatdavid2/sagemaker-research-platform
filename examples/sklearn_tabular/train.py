import argparse, os, glob, json, time
import pandas as pd, joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

p=argparse.ArgumentParser()
p.add_argument("--epochs",type=int,default=1)
p.add_argument("--batch-size",type=int,default=64)
a=p.parse_args()

data_dir=os.getenv("SM_CHANNEL_TRAINING") or os.getenv("LOCAL_DATA_DIR") or ""
files=glob.glob(os.path.join(data_dir,"*.csv")) if data_dir else []
if files:
    df=pd.read_csv(files[0])
    if "label" not in df.columns: raise ValueError("CSV must contain label column")
    X=df.drop(columns=["label"]); y=df["label"]
    source="uploaded CSV"
else:
    d=load_breast_cancer(as_frame=True)
    X=d.data; y=d.target
    source="Wisconsin Breast Cancer"

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=7,stratify=y)
m=RandomForestClassifier(n_estimators=120,n_jobs=-1,random_state=7)
start=time.time(); m.fit(Xtr,ytr); elapsed=time.time()-start
acc=accuracy_score(yte,m.predict(Xte))
print("METRICS",json.dumps({"framework":"sklearn","dataset":source,"accuracy":acc,"elapsed_seconds":elapsed}))
model_dir=os.getenv("SM_MODEL_DIR") or os.getenv("LOCAL_MODEL_DIR") or "/tmp/model"
os.makedirs(model_dir,exist_ok=True)
joblib.dump(m,os.path.join(model_dir,"model.joblib"))
