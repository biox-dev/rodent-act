"""Train the 0.5 s immobile/mobile classifier with XGBoost.

Input CSV comes from ``extract_features.py`` plus a ``label`` column:
``1 = immobile`` (TST score), ``0 = mobile``.

Usage
-----
    python analytics/train_xgb.py data/windows_labeled.csv models/tst_clf.json \
        --cv 5 --groups track_id

``--groups track_id`` keeps all windows of one animal in the same fold, which
matters here: windows of the same recording are strongly correlated and a
random split would report a much too optimistic accuracy.

Labelling tips
--------------
* Label **whole tracks** from a few trials by eye, then let the model
  interpolate; window-level labels are expensive and noisy.
* Keep an entire trial (video) out of training to measure generalisation.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from features import FEATURE_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out", help="output XGBoost JSON model")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--groups", default="track_id", help="column used to group folds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import xgboost as xgb
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as e:  # pragma: no cover
        sys.exit(f"missing dependency: {e}\n  pip install xgboost scikit-learn pandas")

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing feature columns: {missing}")
    if "label" not in df.columns:
        sys.exit("CSV needs a `label` column (1 = immobile, 0 = mobile)")

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    groups = df[args.groups].to_numpy() if args.groups in df.columns else None

    if len(np.unique(y)) < 2:
        sys.exit("only one class present - cannot train")

    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=4,
        eta=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=1.0,
        tree_method="hist",
        seed=args.seed,
    )

    # --- cross validation -----------------------------------------------
    splitter = (
        StratifiedGroupKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        if groups is not None
        else None
    )
    if splitter is not None:
        aucs = []
        for k, (tr, va) in enumerate(splitter.split(X, y, groups), 1):
            dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=FEATURE_NAMES)
            dva = xgb.DMatrix(X[va], label=y[va], feature_names=FEATURE_NAMES)
            bst = xgb.train(params, dtr, num_boost_round=400, evals=[(dva, "val")],
                            early_stopping_rounds=40, verbose_eval=False)
            p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
            auc = roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else float("nan")
            acc = float(((p > 0.5).astype(int) == y[va]).mean())
            aucs.append(auc)
            print(f"fold {k}: n_val={len(va):4d}  auc={auc:.3f}  acc={acc:.3f}")
        print(f"CV AUC = {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")

    # --- final model on everything ---------------------------------------
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    rounds = 400
    bst = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)

    p = bst.predict(dtrain)
    print("\ntrain report (sanity check only):")
    print(classification_report(y, (p > 0.5).astype(int), digits=3))

    importances = bst.get_score(importance_type="gain")
    print("\ntop features by gain:")
    for name, gain in sorted(importances.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:<22} {gain:8.1f}")

    bst.save_model(args.out)

    # side-car with the exact feature order - the Rust loader verifies it
    meta_path = args.out.rsplit(".", 1)[0] + ".features.json"
    with open(meta_path, "w") as f:
        json.dump({"features": FEATURE_NAMES, "objective": params["objective"]}, f, indent=2)
    print(f"\nwrote {args.out} and {meta_path}")


if __name__ == "__main__":
    main()
