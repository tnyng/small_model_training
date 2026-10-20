from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import hstack, csr_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--splits", type=Path, default=Path("experiments/prm_training/splits_v2"))
args = ap.parse_args()
SPL = args.splits

def load(split):
    return [json.loads(l) for l in open(SPL / f"{split}.jsonl") if l.strip()]

def fit_eval(name, Xtr, ytr, Xdv, ydv, prior):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xdv)[:, 1]
    auc = roc_auc_score(ydv, p); ap = average_precision_score(ydv, p)
    print(f"  {name:38s} AUC {auc:.4f}   AP {ap:.4f}   (AP基线={prior:.4f}, lift {ap/prior:.2f}x)")
    return auc, ap

def run(target):
    tr, dv = load("train"), load("dev")
    mk = f"mask_{target}"
    tr = [r for r in tr if r[mk]]; dv = [r for r in dv if r[mk]]
    ytr = np.array([r[target] for r in tr]); ydv = np.array([r[target] for r in dv])
    prior = ydv.mean()
    print(f"\n{'='*74}\n目标 {target}   train {len(tr):,} ({ytr.mean():.3%} 正)   dev {len(dv):,} ({prior:.3%} 正)\n{'='*74}")
    print(f"  {'常数预测 (多数类)':36s} AUC 0.5000   AP {prior:.4f}   (= 基线)")

    # 1) 仅长度
    f = lambda rs: np.array([[len(r['sentence']), len(r['sentence'].split())] for r in rs], dtype=float)
    fit_eval("仅句子长度", f(tr), ytr, f(dv), ydv, prior)

    # 2) 仅句子 TF-IDF  <-- 消融: 模型看不到原文
    v = TfidfVectorizer(max_features=60000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
    A = v.fit_transform([r["sentence"] for r in tr]); B = v.transform([r["sentence"] for r in dv])
    a1, p1 = fit_eval("仅句子 TF-IDF  [无原文]", A, ytr, B, ydv, prior)

    # 3) 句子 + 原文 TF-IDF
    v2 = TfidfVectorizer(max_features=60000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
    src_tr = [r["classification_input"] for r in tr]; src_dv = [r["classification_input"] for r in dv]
    C = v2.fit_transform(src_tr); D = v2.transform(src_dv)
    a2, p2 = fit_eval("句子 + 原文 TF-IDF", hstack([A, C]).tocsr(), ytr, hstack([B, D]).tocsr(), ydv, prior)

    # 4) 词重叠特征 (句子与原文的 lexical overlap)
    def ov(rs):
        out=[]
        for r in rs:
            s=set(r['sentence'].lower().split()); t=set(r['classification_input'].lower().split())
            inter=len(s&t); out.append([inter, inter/max(1,len(s)), inter/max(1,len(t))])
        return np.array(out,dtype=float)
    fit_eval("仅词重叠(3维)", ov(tr), ytr, ov(dv), ydv, prior)

    print(f"\n  >>> 消融判据: 加入原文带来的 AP 增量 = {p2-p1:+.4f}  ({100*(p2-p1)/max(p1,1e-9):+.1f}%)")
    if p2 - p1 < 0.02 * p1:
        print("      ⚠️  原文几乎没有贡献 —— 标签可由句子本身预测, 不是在学 grounding")
    else:
        print("      ✅  原文有实质贡献 —— 任务确实依赖原文")

for t in ("G", "F"):
    run(t)
