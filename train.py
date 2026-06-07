#!/usr/bin/env python3
"""
NIDS Training Pipeline — NSL-KDD
Models: Random Forest, Gradient Boosting, PyTorch DeepMLP, PyTorch LSTM
Outputs: versioned models, confusion matrices, ROC, PR curves, feature importance
"""

import os, sys, json, pickle, warnings, time, hashlib
import numpy as np
import pandas as pd
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_model import DeepMLP, LSTMClassifier, FocalLoss, get_class_weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

ATTACK_MAP = {
    'normal':'Normal',
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS','worm':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L',
    'xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'mailbomb':'U2R','ps':'U2R','sqlattack':'U2R','xterm':'U2R'
}

URLS = {
    'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt',
    'test':  'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt'
}

def get_version():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"v_{ts}_{hashlib.md5(ts.encode()).hexdigest()[:6]}"

VERSION   = get_version()
MODEL_DIR = Path('models') / VERSION
PLOT_DIR  = Path('plots')  / VERSION
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Version: {VERSION}")

def download_data():
    Path('data').mkdir(exist_ok=True)
    for split, url in URLS.items():
        p = Path(f'data/nsl_kdd_{split}.csv')
        if not p.exists():
            print(f"  Downloading {split}...")
            urllib.request.urlretrieve(url, p)
        else:
            print(f"  Cached {split}.")
    return 'data/nsl_kdd_train.csv', 'data/nsl_kdd_test.csv'

def load_and_preprocess(train_path, test_path):
    df_tr = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_te = pd.read_csv(test_path,  header=None, names=COLUMNS)
    for df in [df_tr, df_te]:
        df.drop('difficulty', axis=1, inplace=True)
        df['attack_category'] = df['label'].map(lambda x: ATTACK_MAP.get(x.strip(), 'Other'))
    print(f"\n  Train: {len(df_tr):,}  Test: {len(df_te):,}")
    for cat, cnt in df_tr['attack_category'].value_counts().items():
        print(f"    {cat:<10} {cnt:>7,}  ({cnt/len(df_tr)*100:.1f}%)")
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([df_tr[col], df_te[col]]))
        df_tr[col] = le.transform(df_tr[col])
        df_te[col] = le.transform(df_te[col])
        encoders[col] = le
    cat_enc = LabelEncoder()
    cat_enc.fit(pd.concat([df_tr['attack_category'], df_te['attack_category']]))
    df_tr['multi_label'] = cat_enc.transform(df_tr['attack_category'])
    df_te['multi_label'] = cat_enc.transform(df_te['attack_category'])
    feat_cols = [c for c in df_tr.columns
                 if c not in ('label','attack_category','multi_label')]
    scaler = StandardScaler()
    scaler.fit(df_tr[feat_cols].values)
    return df_tr, df_te, feat_cols, cat_enc, encoders, scaler

def compute_metrics(y_true, y_pred, y_proba, classes, prefix):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    rpt  = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    print(f"\n  {prefix}: Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f}")
    print(rpt)
    return dict(accuracy=acc, f1=f1, precision=prec, recall=rec,
                confusion_matrix=cm.tolist(), report=rpt)

DARK = '#0d1117'
COLORS = ['#00ff88','#ff3b5c','#ff9f0a','#bf5af2','#0a84ff','#ff6b35']

def _dark_ax(fig, ax):
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#333')

def plot_confusion_matrix(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(7,5.5))
    _dark_ax(fig, ax)
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha='right', color='white', fontsize=9)
    ax.set_yticklabels(classes, color='white', fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=8,
                    color='white' if cm[i,j]<cm.max()/2 else 'black')
    ax.set_xlabel('Predicted',color='white'); ax.set_ylabel('True',color='white')
    ax.set_title(title,color='#00ff88',fontsize=12,pad=10)
    plt.tight_layout(); plt.savefig(path,dpi=120,bbox_inches='tight',facecolor=DARK); plt.close()
    print(f"  → {path}")

def plot_roc_curves(y_true, y_proba, classes, title, path):
    n = len(classes); y_bin = label_binarize(y_true, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(7,5.5)); _dark_ax(fig, ax)
    for i,cls in enumerate(classes):
        if y_bin[:,i].sum()==0: continue
        fpr,tpr,_ = roc_curve(y_bin[:,i],y_proba[:,i])
        ax.plot(fpr,tpr,color=COLORS[i%len(COLORS)],
                label=f'{cls} (AUC={auc(fpr,tpr):.3f})',linewidth=1.5)
    ax.plot([0,1],[0,1],'w--',alpha=0.3)
    ax.set_xlabel('FPR',color='white'); ax.set_ylabel('TPR',color='white')
    ax.set_title(title,color='#00ff88',fontsize=12)
    ax.legend(fontsize=8,facecolor='#1a1f2e',labelcolor='white')
    plt.tight_layout(); plt.savefig(path,dpi=120,bbox_inches='tight',facecolor=DARK); plt.close()
    print(f"  → {path}")

def plot_pr_curves(y_true, y_proba, classes, title, path):
    n = len(classes); y_bin = label_binarize(y_true, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(7,5.5)); _dark_ax(fig, ax)
    for i,cls in enumerate(classes):
        if y_bin[:,i].sum()==0: continue
        p,r,_ = precision_recall_curve(y_bin[:,i],y_proba[:,i])
        ap = average_precision_score(y_bin[:,i],y_proba[:,i])
        ax.plot(r,p,color=COLORS[i%len(COLORS)],label=f'{cls} (AP={ap:.3f})',linewidth=1.5)
    ax.set_xlabel('Recall',color='white'); ax.set_ylabel('Precision',color='white')
    ax.set_title(title,color='#00ff88',fontsize=12)
    ax.legend(fontsize=8,facecolor='#1a1f2e',labelcolor='white')
    plt.tight_layout(); plt.savefig(path,dpi=120,bbox_inches='tight',facecolor=DARK); plt.close()
    print(f"  → {path}")

def plot_feature_importance(feat_cols, importances, path):
    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(8,6)); _dark_ax(fig, ax)
    ax.barh([feat_cols[i] for i in idx], importances[idx], color='#0a84ff', alpha=0.85)
    ax.set_xlabel('Importance',color='white')
    ax.set_title('Top-20 Feature Importances (RF)',color='#00ff88',fontsize=12)
    plt.tight_layout(); plt.savefig(path,dpi=120,bbox_inches='tight',facecolor=DARK); plt.close()
    print(f"  → {path}")

def plot_training_curve(tl, vl, va, title, path):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4))
    for ax in [ax1,ax2]: _dark_ax(fig,ax)
    ax1.plot(tl,color='#0a84ff',label='Train'); ax1.plot(vl,color='#ff3b5c',label='Val')
    ax1.set_title('Loss',color='#00ff88'); ax1.legend(facecolor='#1a1f2e',labelcolor='white')
    ax1.set_xlabel('Epoch',color='white')
    ax2.plot(va,color='#00ff88'); ax2.set_title('Val Accuracy',color='#00ff88')
    ax2.set_xlabel('Epoch',color='white')
    fig.suptitle(title,color='white',fontsize=12)
    plt.tight_layout(); plt.savefig(path,dpi=120,bbox_inches='tight',facecolor=DARK); plt.close()
    print(f"  → {path}")

def train_sklearn(X_tr,y_tr,X_te,y_te,feat_cols,class_names,scaler):
    results={}
    print("\n  [sklearn] Random Forest")
    rf = RandomForestClassifier(n_estimators=300,n_jobs=-1,max_depth=25,
                                 random_state=42,class_weight='balanced')
    rf.fit(X_tr,y_tr)
    y_pred=rf.predict(X_te); y_proba=rf.predict_proba(X_te)
    m=compute_metrics(y_te,y_pred,y_proba,class_names,'RF Multi')
    plot_confusion_matrix(np.array(m['confusion_matrix']),class_names,
        'RF — Confusion Matrix',PLOT_DIR/'rf_confusion.png')
    plot_roc_curves(y_te,y_proba,class_names,'RF — ROC',PLOT_DIR/'rf_roc.png')
    plot_pr_curves(y_te,y_proba,class_names,'RF — PR',PLOT_DIR/'rf_pr.png')
    plot_feature_importance(feat_cols,rf.feature_importances_,PLOT_DIR/'rf_features.png')
    mp=MODEL_DIR/'rf_multi.pkl'
    with open(mp,'wb') as f:
        pickle.dump({'model':rf,'scaler':scaler,'feat_cols':feat_cols,'classes':list(class_names)},f)
    with open(MODEL_DIR/'scaler.pkl','wb') as f: pickle.dump(scaler,f)
    results['rf_multi']={**m,'path':str(mp),'framework':'sklearn','type':'multi',
                         'classes':list(class_names),
                         'feature_importances':list(zip(feat_cols,rf.feature_importances_.tolist()))}

    print("\n  [sklearn] Gradient Boosting (binary)")
    normal_idx=list(class_names).index('Normal') if 'Normal' in list(class_names) else 0
    y_bin_tr=(y_tr!=normal_idx).astype(int); y_bin_te=(y_te!=normal_idx).astype(int)
    gb=GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,random_state=42)
    gb.fit(X_tr,y_bin_tr)
    y_pb=gb.predict(X_te); y_proba_b=gb.predict_proba(X_te)
    m_b=compute_metrics(y_bin_te,y_pb,y_proba_b,['Normal','Attack'],'GB Binary')
    mp_b=MODEL_DIR/'gb_binary.pkl'
    with open(mp_b,'wb') as f:
        pickle.dump({'model':gb,'scaler':scaler,'feat_cols':feat_cols,'classes':['Normal','Attack']},f)
    results['gb_binary']={**m_b,'path':str(mp_b),'framework':'sklearn','type':'binary','classes':['Normal','Attack']}
    return results

def train_torch_model(model,name,X_tr,y_tr,X_te,y_te,n_classes,class_names,epochs=40,batch=512,lr=1e-3):
    model=model.to(DEVICE)
    weights=get_class_weights(y_tr,n_classes,DEVICE)
    criterion=FocalLoss(gamma=2.0,weight=weights)
    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    Xtr=torch.FloatTensor(X_tr).to(DEVICE); ytr=torch.LongTensor(y_tr).to(DEVICE)
    Xte=torch.FloatTensor(X_te).to(DEVICE); yte=torch.LongTensor(y_te).to(DEVICE)
    dl=DataLoader(TensorDataset(Xtr,ytr),batch_size=batch,shuffle=True,drop_last=True)
    tl,vl,va=[],[],[]; best=0; ckpt=MODEL_DIR/f'{name}_best.pt'
    is_lstm=hasattr(model,'lstm')
    print(f"\n  [PyTorch] {name} — {epochs} epochs")
    for ep in range(epochs):
        model.train(); ep_loss=0
        for xb,yb in dl:
            optimizer.zero_grad()
            inp=xb.unsqueeze(1) if is_lstm else xb
            loss=criterion(model(inp),yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); ep_loss+=loss.item()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            inp_te=Xte.unsqueeze(1) if is_lstm else Xte
            logits=model(inp_te)
            vl_=criterion(logits,yte).item()
            acc_=(logits.argmax(1)==yte).float().mean().item()
        tl.append(ep_loss/len(dl)); vl.append(vl_); va.append(acc_)
        if acc_>best:
            best=acc_
            torch.save({'epoch':ep,'state_dict':model.state_dict(),'val_acc':acc_,
                        'n_classes':n_classes,'classes':list(class_names)},ckpt)
        if (ep+1)%10==0:
            print(f"    Ep{ep+1:>3}/{epochs} loss={tl[-1]:.4f} val={vl_:.4f} acc={acc_:.4f}")
    plot_training_curve(tl,vl,va,f'{name} Training',PLOT_DIR/f'{name}_training.png')
    ckpt_data=torch.load(ckpt,map_location=DEVICE)
    model.load_state_dict(ckpt_data['state_dict']); model.eval()
    with torch.no_grad():
        inp_te=Xte.unsqueeze(1) if is_lstm else Xte
        logits=model(inp_te)
        y_pred=logits.argmax(1).cpu().numpy()
        y_proba=torch.softmax(logits,1).cpu().numpy()
    m=compute_metrics(y_te,y_pred,y_proba,class_names,name)
    plot_confusion_matrix(np.array(m['confusion_matrix']),class_names,
        f'{name} — Confusion Matrix',PLOT_DIR/f'{name}_confusion.png')
    plot_roc_curves(y_te,y_proba,class_names,f'{name} — ROC',PLOT_DIR/f'{name}_roc.png')
    plot_pr_curves(y_te,y_proba,class_names,f'{name} — PR',PLOT_DIR/f'{name}_pr.png')
    return {**m,'path':str(ckpt),'framework':'pytorch','type':'multi',
            'classes':list(class_names),'best_val_acc':best}

def main():
    print("="*65)
    print(f"  NIDS TRAINING  —  {VERSION}")
    print("="*65)

    print("\n[1/5] Data...")
    tr,te=download_data()

    print("\n[2/5] Preprocessing...")
    df_tr,df_te,feat_cols,cat_enc,encoders,scaler=load_and_preprocess(tr,te)
    class_names=cat_enc.classes_; n_classes=len(class_names); n_feat=len(feat_cols)
    X_tr_raw=df_tr[feat_cols].values; X_te_raw=df_te[feat_cols].values
    y_tr=df_tr['multi_label'].values;  y_te=df_te['multi_label'].values
    X_tr_sc=scaler.transform(X_tr_raw).astype(np.float32)
    X_te_sc=scaler.transform(X_te_raw).astype(np.float32)

    all_res={}

    print("\n[3/5] Sklearn models...")
    all_res.update(train_sklearn(X_tr_raw,y_tr,X_te_raw,y_te,feat_cols,class_names,scaler))

    print("\n[4/5] PyTorch models...")
    all_res['deep_mlp']=train_torch_model(
        DeepMLP(n_feat,n_classes,256,0.3),'deep_mlp',
        X_tr_sc,y_tr,X_te_sc,y_te,n_classes,class_names,epochs=40)
    all_res['lstm']=train_torch_model(
        LSTMClassifier(n_feat,n_classes,128,2,0.3),'lstm',
        X_tr_sc,y_tr,X_te_sc,y_te,n_classes,class_names,epochs=30)

    print("\n[5/5] Saving metadata...")
    best=max(all_res,key=lambda k:all_res[k]['f1'])
    safe={n:{k:v for k,v in i.items() if k not in ('confusion_matrix','report','feature_importances')}
          for n,i in all_res.items()}
    meta={
        'version':VERSION, 'created_at':datetime.now().isoformat(),
        'best_model':best, 'models':safe,
        'feature_cols':feat_cols,
        'top_features':sorted(all_res['rf_multi']['feature_importances'],
                               key=lambda x:x[1],reverse=True)[:20],
        'class_names':list(class_names),
        'n_classes':n_classes, 'n_features':n_feat,
        'plots_dir':str(PLOT_DIR),
    }
    mp=MODEL_DIR/'metadata.json'
    with open(mp,'w') as f: json.dump(meta,f,indent=2)
    with open('models/latest.json','w') as f:
        json.dump({'version':VERSION,'model_dir':str(MODEL_DIR),'metadata':str(mp)},f,indent=2)
    with open('models/registry.jsonl','a') as f:
        f.write(json.dumps({'version':VERSION,'created_at':meta['created_at'],'best_model':best,
            'metrics':{k:{'f1':v['f1'],'accuracy':v['accuracy']} for k,v in safe.items()}})+'\n')

    print("\n"+"="*65+"  SUMMARY")
    for n,i in all_res.items():
        print(f"  {n:<15} Acc={i['accuracy']:.4f} F1={i['f1']:.4f} [{i['framework']}]"
              +(" ★" if n==best else ""))
    print(f"\n  Plots  → {PLOT_DIR}/")
    print(f"  Models → {MODEL_DIR}/")

if __name__=='__main__':
    main()
