#!/usr/bin/env python3
"""
NIDS Inference Server — FastAPI v2
- Multi-model prediction + runtime model swap
- Threshold-based alerting engine
- Retraining trigger (background subprocess)
- PCAP ingestion via scapy
- WebSocket live feed
- Feedback/labelling for retraining
"""

import json, pickle, time, asyncio, subprocess, sys, traceback
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Optional, Dict, List, Any

import numpy as np
import torch
from torch_model import DeepMLP, LSTMClassifier

from fastapi import (FastAPI, HTTPException, WebSocket,
                     WebSocketDisconnect, BackgroundTasks, UploadFile, File)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="NIDS API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS: Dict[str, Any] = {}
META: Dict = {}
CURRENT_VERSION: str = ""
ALERT_RULES: List[Dict] = []
ALERT_HISTORY: deque = deque(maxlen=500)
FEEDBACK_STORE: List[Dict] = []
TRAFFIC_WINDOW: deque = deque(maxlen=600)
CONNECTED_WS: List[WebSocket] = []


def load_version(version="latest"):
    global META, CURRENT_VERSION, MODELS, ALERT_RULES
    MODELS.clear()
    if version == "latest":
        lp = Path("models/latest.json")
        if not lp.exists():
            raise RuntimeError("No models found. Run train.py first.")
        ptr = json.loads(lp.read_text())
        meta_path = Path(ptr["metadata"])
        CURRENT_VERSION = ptr["version"]
    else:
        meta_path = Path("models") / version / "metadata.json"
        CURRENT_VERSION = version
    META = json.loads(meta_path.read_text())
    model_dir = meta_path.parent
    n_feat = META["n_features"]
    n_cls  = META["n_classes"]
    classes = META["class_names"]

    scaler = None
    sp = model_dir / "scaler.pkl"
    if sp.exists():
        with open(sp,"rb") as f: scaler = pickle.load(f)

    for name, info in META["models"].items():
        path = Path(info["path"])
        if not path.exists(): continue
        if info["framework"] == "sklearn":
            with open(path,"rb") as f: obj = pickle.load(f)
            if scaler and obj.get("scaler") is None: obj["scaler"] = scaler
            MODELS[name] = obj
        elif info["framework"] == "pytorch":
            ckpt = torch.load(path, map_location=DEVICE)
            m = DeepMLP(n_feat,n_cls,256,0.3) if ("mlp" in name or "deep" in name) \
                else LSTMClassifier(n_feat,n_cls,128,2,0.3)
            m.load_state_dict(ckpt["state_dict"]); m.eval().to(DEVICE)
            MODELS[name] = {"model":m,"scaler":scaler,"feat_cols":META["feature_cols"],
                            "classes":classes,"framework":"pytorch"}

    ALERT_RULES = [
        {"id":"dos_spike",  "category":"DoS",  "window_sec":10,"threshold":0.30,"severity":"critical"},
        {"id":"probe_spike","category":"Probe","window_sec":30,"threshold":0.20,"severity":"high"},
        {"id":"r2l_any",    "category":"R2L",  "window_sec":60,"threshold":0.05,"severity":"high"},
        {"id":"u2r_any",    "category":"U2R",  "window_sec":60,"threshold":0.02,"severity":"critical"},
    ]
    print(f"Loaded {len(MODELS)} models [{CURRENT_VERSION}]")


@app.on_event("startup")
def startup():
    try: load_version("latest")
    except Exception as e: print(f"WARNING: {e}")


class PredictRequest(BaseModel):
    features: Dict[str, float]
    model: Optional[str] = "rf_multi"

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    is_attack: bool
    model_used: str
    alerts_fired: List[str] = []

class FeedbackRequest(BaseModel):
    sample_id: str
    features: Dict[str, float]
    predicted_label: str
    true_label: str
    notes: Optional[str] = ""

class AlertRuleModel(BaseModel):
    id: str; category: str; window_sec: int; threshold: float; severity: str


def _predict_one(model_name, features):
    if model_name not in MODELS:
        raise HTTPException(400, f"Unknown model: {model_name}. Have: {list(MODELS.keys())}")
    obj = MODELS[model_name]
    feat_cols = obj["feat_cols"]
    X = np.array([[features.get(c,0.0) for c in feat_cols]])
    if obj.get("framework") == "pytorch":
        scaler = obj["scaler"]; model = obj["model"]; classes = obj["classes"]
        if scaler: X = scaler.transform(X)
        Xt = torch.FloatTensor(X).to(DEVICE)
        if hasattr(model,"lstm"): Xt = Xt.unsqueeze(1)
        with torch.no_grad():
            proba = torch.softmax(model(Xt),1).cpu().numpy()[0]
        pred_idx = int(proba.argmax())
        return classes[pred_idx], float(proba.max()), dict(zip(classes,proba.tolist()))
    else:
        scaler = obj["scaler"]; model = obj["model"]; classes = obj["classes"]
        if scaler: X = scaler.transform(X)
        pred_idx = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        pred_label = classes[pred_idx]
        return pred_label, float(proba.max()), dict(zip([str(c) for c in classes],proba.tolist()))


def _check_alerts(prediction, confidence):
    now = datetime.utcnow()
    TRAFFIC_WINDOW.append({"time":now,"prediction":prediction,"confidence":confidence})
    fired = []
    for rule in ALERT_RULES:
        cutoff = now - timedelta(seconds=rule["window_sec"])
        win = [p for p in TRAFFIC_WINDOW if p["time"] >= cutoff]
        if not win: continue
        rate = sum(1 for p in win if p["prediction"]==rule["category"]) / len(win)
        if rate >= rule["threshold"]:
            alert = {"rule_id":rule["id"],"category":rule["category"],"rate":round(rate,3),
                     "threshold":rule["threshold"],"severity":rule["severity"],
                     "timestamp":now.isoformat()}
            ALERT_HISTORY.append(alert)
            fired.append(rule["id"])
    return fired


@app.get("/")
def root():
    return {"status":"ok","version":CURRENT_VERSION,"models":list(MODELS.keys())}

@app.get("/metadata")
def metadata(): return META

@app.get("/models")
def list_models():
    return {"current_version":CURRENT_VERSION,"loaded":list(MODELS.keys()),
            "metrics":{k:{"f1":v["f1"],"accuracy":v["accuracy"]}
                       for k,v in META.get("models",{}).items()}}

@app.post("/models/load/{version}")
def load_model_version(version: str):
    try: load_version(version); return {"status":"ok","version":CURRENT_VERSION}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/models/registry")
def version_registry():
    reg = Path("models/registry.jsonl")
    if not reg.exists(): return []
    return [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    pred, conf, probs = _predict_one(req.model, req.features)
    alerts = _check_alerts(pred, conf)
    if CONNECTED_WS:
        msg = json.dumps({"type":"packet","prediction":pred,"confidence":conf,
                          "is_attack":pred!="Normal","alerts":alerts,
                          "ts":datetime.utcnow().isoformat()})
        for ws in list(CONNECTED_WS):
            try: await ws.send_text(msg)
            except: CONNECTED_WS.remove(ws)
    return PredictResponse(prediction=pred,confidence=conf,probabilities=probs,
                           is_attack=pred!="Normal",model_used=req.model,alerts_fired=alerts)

@app.post("/predict/batch")
async def predict_batch(items: List[PredictRequest]):
    return [await predict(r) for r in items]

@app.get("/alerts")
def get_alerts(limit: int = 50): return list(ALERT_HISTORY)[-limit:]

@app.get("/alerts/rules")
def get_rules(): return ALERT_RULES

@app.post("/alerts/rules")
def add_rule(rule: AlertRuleModel):
    ALERT_RULES.append(rule.dict()); return {"status":"added","rules":ALERT_RULES}

@app.delete("/alerts/rules/{rule_id}")
def delete_rule(rule_id: str):
    global ALERT_RULES
    ALERT_RULES = [r for r in ALERT_RULES if r["id"]!=rule_id]
    return {"status":"deleted"}

@app.post("/feedback")
def submit_feedback(fb: FeedbackRequest):
    record = {**fb.dict(),"submitted_at":datetime.utcnow().isoformat()}
    FEEDBACK_STORE.append(record)
    fb_path = Path("logs/feedback.jsonl"); fb_path.parent.mkdir(exist_ok=True)
    with open(fb_path,"a") as f: f.write(json.dumps(record)+"\n")
    return {"status":"recorded","total_feedback":len(FEEDBACK_STORE)}

@app.get("/feedback")
def get_feedback(): return {"count":len(FEEDBACK_STORE),"samples":FEEDBACK_STORE[-20:]}

@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    def _retrain():
        print("[Retrain] Starting...")
        result = subprocess.run([sys.executable,"train.py"],capture_output=True,text=True)
        if result.returncode == 0:
            load_version("latest")
            print(f"[Retrain] Done. Version: {CURRENT_VERSION}")
        else:
            print(f"[Retrain] FAILED:\n{result.stderr}")
    background_tasks.add_task(_retrain)
    return {"status":"retraining_started","feedback_samples":len(FEEDBACK_STORE)}

@app.post("/ingest/pcap")
async def ingest_pcap(file: UploadFile=File(...), model: str="rf_multi"):
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        raise HTTPException(501,"Install scapy: pip install scapy")
    content = await file.read()
    tmp = Path(f"/tmp/{file.filename}"); tmp.write_bytes(content)
    try: packets = rdpcap(str(tmp))
    except Exception as e: raise HTTPException(400, f"Bad PCAP: {e}")
    flows = defaultdict(list)
    for pkt in packets:
        if not pkt.haslayer(IP): continue
        ip = pkt[IP]
        sport = pkt[TCP].sport if pkt.haslayer(TCP) else (pkt[UDP].sport if pkt.haslayer(UDP) else 0)
        dport = pkt[TCP].dport if pkt.haslayer(TCP) else (pkt[UDP].dport if pkt.haslayer(UDP) else 0)
        flows[(ip.src,ip.dst,sport,dport,ip.proto)].append(pkt)
    results = []
    for fk, pkts in list(flows.items())[:200]:
        sb = sum(len(p) for p in pkts)
        feats = {"duration":float(pkts[-1].time-pkts[0].time) if len(pkts)>1 else 0,
                 "protocol_type":float(fk[4]),"src_bytes":float(sb),"dst_bytes":float(sb//2),
                 "count":float(len(pkts)),"serror_rate":0.0,"same_srv_rate":1.0,
                 "dst_host_count":float(len(flows))}
        pred, conf, probs = _predict_one(model, feats)
        results.append({"flow":f"{fk[0]}:{fk[2]} → {fk[1]}:{fk[3]}","packets":len(pkts),
                        "prediction":pred,"confidence":round(conf,4),"is_attack":pred!="Normal"})
    attacks = sum(1 for r in results if r["is_attack"])
    return {"total_flows":len(results),"attacks_detected":attacks,
            "attack_rate":round(attacks/max(len(results),1),3),"flows":results}

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept(); CONNECTED_WS.append(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in CONNECTED_WS: CONNECTED_WS.remove(websocket)

@app.get("/stats")
def live_stats():
    now = datetime.utcnow()
    last60 = [p for p in TRAFFIC_WINDOW if p["time"] >= now-timedelta(seconds=60)]
    by_cat = defaultdict(int)
    for p in last60: by_cat[p["prediction"]] += 1
    total = len(last60) or 1
    return {"total_processed":len(TRAFFIC_WINDOW),
            "last_60s":{k:{"count":v,"rate":round(v/total,3)} for k,v in by_cat.items()},
            "active_alerts":len([a for a in ALERT_HISTORY
                if datetime.fromisoformat(a["timestamp"]) >= now-timedelta(minutes=5)]),
            "version":CURRENT_VERSION}
