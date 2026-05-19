"""
AudioMark Annotator — Standalone Training Script
Run this after annotating your files in the browser tool.

Usage:
    python train.py                         # SVM + MFCC (default)
    python train.py --model svm             # Explicit
    python train.py --model yamnet          # YAMNet (needs tensorflow)
    python train.py --features pcen         # PCEN features (better for noisy audio)
    python train.py --model svm --features pcen  # Best combo for factory data
    python train.py --predict audio.wav     # Predict on a file after training

Models: svm, random_forest, knn, gradient_boost, mlp, logistic, naive_bayes, yamnet
"""
import os,sys,json,pickle,argparse,time
from pathlib import Path
import numpy as np

BASE    = Path(__file__).parent
UPLOADS = BASE/"data"/"uploads"
ANNOTS  = BASE/"data"/"annotations"
SLICED  = BASE/"data"/"sliced"
MODELS  = BASE/"data"/"models"
MODELS.mkdir(exist_ok=True)

print("\n"+"="*56)
print("  AudioMark Annotator — Training Script")
print("  Mahindra Connector Lock Detection")
print("="*56+"\n")

def check_annotations():
    ann_files=list(ANNOTS.glob("*.json"))
    if not ann_files:
        print("❌  No annotations found. Annotate files in the browser tool first.")
        sys.exit(1)
    label_counts={}; total=0
    for af in ann_files:
        for seg in json.loads(af.read_text()).get("segments",[]):
            lbl=seg["label"]; label_counts[lbl]=label_counts.get(lbl,0)+1; total+=1
    print(f"📋  {len(ann_files)} annotated files  →  {total} total segments")
    for lbl,cnt in sorted(label_counts.items(),key=lambda x:-x[1]):
        ok="✅" if cnt>=20 else "⚠️ "
        bar="█"*min(30,cnt)
        print(f"      {ok}  {lbl:<30} {cnt:>4} segments  {bar}")
    if len(label_counts)<2:
        print("\n❌  Need at least 2 labels."); sys.exit(1)
    return label_counts

def slice_audio():
    try: import librosa,soundfile as sf
    except ImportError: print("❌  pip install librosa soundfile"); sys.exit(1)
    import shutil
    if SLICED.exists(): shutil.rmtree(SLICED)
    SLICED.mkdir()
    stats={"clips":0,"by_label":{}}
    for af in ANNOTS.glob("*.json"):
        data=json.loads(af.read_text()); src=UPLOADS/data["filename"]
        if not src.exists(): print(f"  ⚠️  {data['filename']} not found — skipping"); continue
        audio,sr=librosa.load(str(src),sr=16000,mono=True)
        for j,seg in enumerate(data.get("segments",[])):
            clip=audio[int(seg["start"]*sr):int(seg["end"]*sr)]
            if len(clip)<100: continue
            lbl=seg["label"]; od=SLICED/lbl; od.mkdir(exist_ok=True)
            sf.write(str(od/f"{Path(data['filename']).stem}_{j:04d}.wav"),clip,sr,subtype="PCM_16")
            stats["clips"]+=1; stats["by_label"][lbl]=stats["by_label"].get(lbl,0)+1
    print(f"\n✂   Sliced {stats['clips']} clips:")
    for lbl,cnt in sorted(stats["by_label"].items(),key=lambda x:-x[1]):
        print(f"      {lbl:<32} {cnt} clips")
    if not stats["clips"]: print("❌  No clips sliced."); sys.exit(1)

def extract_mfcc():
    import librosa
    label_set=sorted(set(p.parent.name for p in SLICED.rglob("*.wav")))
    label2idx={l:i for i,l in enumerate(label_set)}
    X,y,skip=[],[],0
    all_wavs=sorted(SLICED.rglob("*.wav"))
    print(f"\n📊  Extracting MFCC features ({len(all_wavs)} clips) ...")
    for wav in all_wavs:
        try:
            audio,sr=librosa.load(str(wav),sr=16000,duration=1.5)
            mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=40)
            chroma=librosa.feature.chroma_stft(y=audio,sr=sr)
            zcr=librosa.feature.zero_crossing_rate(y=audio)
            sc=librosa.feature.spectral_centroid(y=audio,sr=sr)
            feat=np.concatenate([np.mean(mfcc,axis=1),np.std(mfcc,axis=1),
                                  np.mean(chroma,axis=1),
                                  [np.mean(zcr),np.std(zcr),np.mean(sc),np.std(sc)]])
            X.append(feat); y.append(label2idx[wav.parent.name])
        except: skip+=1
    if skip: print(f"  ⚠️  Skipped {skip} clips")
    print(f"  ✅  {len(X)} vectors × {len(X[0]) if X else 0} features")
    return np.array(X),np.array(y),label_set,label2idx

def extract_pcen():
    import librosa
    label_set=sorted(set(p.parent.name for p in SLICED.rglob("*.wav")))
    label2idx={l:i for i,l in enumerate(label_set)}
    X,y,skip=[],[],0
    all_wavs=sorted(SLICED.rglob("*.wav"))
    print(f"\n⭐  Extracting PCEN features ({len(all_wavs)} clips) ...")
    for wav in all_wavs:
        try:
            audio,sr=librosa.load(str(wav),sr=16000,duration=1.5)
            hop=256
            mel=librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=128,fmax=8000,hop_length=hop,n_fft=1024)
            pcen=librosa.pcen(mel*(2**31),sr=sr,hop_length=hop,gain=0.8,bias=10,power=0.25,time_constant=0.4,eps=1e-6)
            delta=librosa.feature.delta(pcen)
            contrast=librosa.feature.spectral_contrast(y=audio,sr=sr,n_fft=1024,hop_length=hop,n_bands=6)
            zcr=librosa.feature.zero_crossing_rate(y=audio)
            feat=np.concatenate([np.mean(pcen,axis=1),np.std(pcen,axis=1),np.mean(delta,axis=1),
                                  np.mean(contrast,axis=1),[float(np.mean(zcr)),float(np.std(zcr))]])
            X.append(feat); y.append(label2idx[wav.parent.name])
        except: skip+=1
    if skip: print(f"  ⚠️  Skipped {skip} clips")
    print(f"  ✅  {len(X)} vectors × {len(X[0]) if X else 0} features")
    return np.array(X),np.array(y),label_set,label2idx

def train_sklearn(X,y,label_set,model_type):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics          import accuracy_score,classification_report
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc=StandardScaler(); X_tr=sc.fit_transform(X_tr); X_te=sc.transform(X_te)
    print(f"\n🧠  Training {model_type.upper()} ...  Train:{len(X_tr)}  Test:{len(X_te)}")
    t0=time.time()
    if model_type=="svm":
        from sklearn.svm import SVC
        clf=SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=42)
    elif model_type=="random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf=RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42)
    elif model_type=="knn":
        from sklearn.neighbors import KNeighborsClassifier
        clf=KNeighborsClassifier(n_neighbors=5)
    elif model_type=="gradient_boost":
        from sklearn.ensemble import GradientBoostingClassifier
        clf=GradientBoostingClassifier(n_estimators=200,random_state=42)
    elif model_type=="mlp":
        from sklearn.neural_network import MLPClassifier
        clf=MLPClassifier(hidden_layer_sizes=(256,128),max_iter=300,early_stopping=True,random_state=42)
    elif model_type=="logistic":
        from sklearn.linear_model import LogisticRegression
        clf=LogisticRegression(max_iter=1000,random_state=42)
    elif model_type=="naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        clf=GaussianNB()
    else: print(f"❌  Unknown model: {model_type}"); sys.exit(1)
    clf.fit(X_tr,y_tr)
    elapsed=round(time.time()-t0,1)
    y_pred=clf.predict(X_te)
    accuracy=round(accuracy_score(y_te,y_pred)*100,1)
    report=classification_report(y_te,y_pred,target_names=label_set,zero_division=0)
    print(f"\n{'='*48}\n  ✅  Accuracy: {accuracy}%   ({elapsed}s)\n{'='*48}\n{report}")
    md={"model":clf,"scaler":sc,"model_type":model_type,"labels":label_set,
        "label2idx":{l:i for i,l in enumerate(label_set)},
        "idx2label":{i:l for i,l in enumerate(label_set)},
        "accuracy":accuracy,"n_features":X.shape[1]}
    with open(MODELS/"model.pkl","wb") as f: pickle.dump(md,f)
    (MODELS/"last_result.json").write_text(json.dumps({"model_type":model_type,"accuracy":accuracy,
        "train_samples":len(X_tr),"test_samples":len(X_te),"labels":label_set,"report":report},indent=2))
    print(f"\n💾  Saved: data/models/model.pkl")
    return accuracy

def train_yamnet(label_set):
    try: import tensorflow as tf; import tensorflow_hub as hub
    except ImportError: print("❌  pip install tensorflow tensorflow-hub"); sys.exit(1)
    import librosa
    print("\n🧠  Training YAMNet (downloading ~13MB model on first run) ...")
    ym=hub.load('https://tfhub.dev/google/yamnet/1')
    all_wavs=sorted(SLICED.rglob("*.wav"))
    l2i={l:i for i,l in enumerate(label_set)}
    embs,labs=[],[]
    for wav in all_wavs:
        try:
            audio,sr=librosa.load(str(wav),sr=16000,mono=True,duration=1.5)
            if len(audio)<1600: audio=np.pad(audio,(0,1600-len(audio)))
            _,emb,_=ym(tf.constant(audio,dtype=tf.float32))
            embs.append(tf.reduce_mean(emb,axis=0).numpy()); labs.append(l2i[wav.parent.name])
        except: pass
    E,L=np.array(embs),np.array(labs)
    from sklearn.model_selection import train_test_split; from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score,classification_report
    X_tr,X_te,y_tr,y_te=train_test_split(E,L,test_size=0.2,random_state=42,stratify=L)
    sc=StandardScaler(); X_tr=sc.fit_transform(X_tr); X_te=sc.transform(X_te)
    n=len(label_set)
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=(1024,)),tf.keras.layers.Dense(256,activation='relu'),tf.keras.layers.Dropout(0.3),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(n,activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    t0=time.time()
    model.fit(X_tr,y_tr,epochs=50,batch_size=32,verbose=0,validation_data=(X_te,y_te),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)])
    y_pred=np.argmax(model.predict(X_te,verbose=0),axis=1)
    accuracy=round(accuracy_score(y_te,y_pred)*100,1)
    report=classification_report(y_te,y_pred,target_names=label_set,zero_division=0)
    print(f"\n{'='*48}\n  ✅  YAMNet Accuracy: {accuracy}%   ({round(time.time()-t0,1)}s)\n{'='*48}\n{report}")
    model.save(str(MODELS/"yamnet_clf.keras"))
    (MODELS/"last_result.json").write_text(json.dumps({"model_type":"yamnet","accuracy":accuracy,
        "train_samples":len(X_tr),"test_samples":len(X_te),"labels":label_set,"report":report},indent=2))
    print(f"\n💾  Saved: data/models/yamnet_clf.keras")
    return accuracy

def quick_predict(audio_file,features="mfcc"):
    pkl=MODELS/"model.pkl"
    if not pkl.exists(): print("❌  No model.pkl — train first"); return
    import librosa
    with open(pkl,"rb") as f: md=pickle.load(f)
    clf=md["model"]; scaler=md["scaler"]; i2l=md["idx2label"]
    audio,sr=librosa.load(audio_file,sr=16000,duration=1.5)
    if features=="pcen":
        hop=256; mel=librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=128,fmax=8000,hop_length=hop,n_fft=1024)
        pcen=librosa.pcen(mel*(2**31),sr=sr,hop_length=hop,gain=0.8,bias=10,power=0.25,time_constant=0.4,eps=1e-6)
        delta=librosa.feature.delta(pcen)
        contrast=librosa.feature.spectral_contrast(y=audio,sr=sr,n_fft=1024,hop_length=hop,n_bands=6)
        zcr=librosa.feature.zero_crossing_rate(y=audio)
        feat=np.concatenate([np.mean(pcen,axis=1),np.std(pcen,axis=1),np.mean(delta,axis=1),np.mean(contrast,axis=1),[float(np.mean(zcr)),float(np.std(zcr))]])
    else:
        mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=40); chroma=librosa.feature.chroma_stft(y=audio,sr=sr)
        zcr=librosa.feature.zero_crossing_rate(y=audio); sc=librosa.feature.spectral_centroid(y=audio,sr=sr)
        feat=np.concatenate([np.mean(mfcc,axis=1),np.std(mfcc,axis=1),np.mean(chroma,axis=1),[np.mean(zcr),np.std(zcr),np.mean(sc),np.std(sc)]])
    fs=scaler.transform([feat])
    if hasattr(clf,'predict_proba'):
        probs=clf.predict_proba(fs)[0]; pi=np.argmax(probs); conf=round(probs[pi]*100,1)
    else: pi=clf.predict(fs)[0]; conf=None
    label=i2l.get(pi,str(pi))
    print(f"\n🔬  Prediction: {label}  ({conf}% confident)")
    if hasattr(clf,'predict_proba'):
        for i,p in enumerate(probs):
            print(f"     {i2l[i]:<30} {round(p*100,1):>5}%  {'█'*int(p*20)}")

def main():
    parser=argparse.ArgumentParser(description="AudioMark Training Script")
    parser.add_argument("--model",default="svm",
        choices=["svm","random_forest","knn","gradient_boost","mlp","logistic","naive_bayes","yamnet"],
        help="Model to train (default: svm)")
    parser.add_argument("--features",default="mfcc",choices=["mfcc","pcen"],
        help="Feature method (default: mfcc, use pcen for noisy audio)")
    parser.add_argument("--predict",default=None,metavar="FILE",
        help="Predict on this audio file after training")
    parser.add_argument("--skip-slice",action="store_true",
        help="Skip slicing (use existing clips)")
    args=parser.parse_args()

    check_annotations()
    if not args.skip_slice: slice_audio()
    else:
        clips=list(SLICED.rglob("*.wav")) if SLICED.exists() else []
        print(f"\n✂   Using {len(clips)} existing clips (--skip-slice)")

    if args.model=="yamnet":
        label_set=sorted(set(p.parent.name for p in SLICED.rglob("*.wav")))
        accuracy=train_yamnet(label_set)
    else:
        X,y,label_set,label2idx=(extract_pcen() if args.features=="pcen" else extract_mfcc())
        accuracy=train_sklearn(X,y,label_set,args.model)

    if args.predict: quick_predict(args.predict,args.features)

    print("\n"+"="*56)
    print(f"  Training complete!  Accuracy: {accuracy}%")
    print(f"  Model: data/models/model.pkl")
    print("="*56)
    print("\n  Next steps:")
    print("  1. Copy data/models/model.pkl to ConnectorCheck app")
    print("  2. Test:  python train.py --predict your_audio.wav")
    print("  3. Low accuracy? Try:  python train.py --model yamnet")
    print("  4. Noisy audio? Try:   python train.py --features pcen\n")

if __name__=="__main__":
    main()
