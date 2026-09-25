import argparse, os, glob, json, time
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

p=argparse.ArgumentParser()
p.add_argument("--epochs",type=int,default=1)
p.add_argument("--batch-size",type=int,default=64)
a=p.parse_args()

data_dir=os.getenv("SM_CHANNEL_TRAINING") or os.getenv("LOCAL_DATA_DIR") or ""
files=glob.glob(os.path.join(data_dir,"*.csv")) if data_dir else []
if files:
    df=pd.read_csv(files[0])
    if "label" not in df.columns: raise ValueError("CSV must contain label column")
    X=df.drop(columns=["label"]).values; y=df["label"].values
    source="uploaded CSV"
else:
    d=load_breast_cancer()
    X=d.data; y=d.target
    source="Wisconsin Breast Cancer"

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=7,stratify=y)
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
m=tf.keras.Sequential([
    tf.keras.layers.Input((Xtr.shape[1],)),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
m.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
start=time.time()
m.fit(Xtr,ytr,epochs=a.epochs,batch_size=a.batch_size,verbose=2)
elapsed=time.time()-start
_,acc=m.evaluate(Xte,yte,verbose=0)
print("METRICS",json.dumps({"framework":"tensorflow","dataset":source,"accuracy":float(acc),"elapsed_seconds":elapsed}))
model_dir=os.getenv("SM_MODEL_DIR") or os.getenv("LOCAL_MODEL_DIR") or "/tmp/model"
os.makedirs(model_dir,exist_ok=True)
m.save(os.path.join(model_dir,"model.keras"))
