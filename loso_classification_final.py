import os
import json
import gc
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from math import floor, log
from collections import defaultdict, Counter

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GroupKFold, HalvingGridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, roc_auc_score

# Riemannian imports (optional)
try:
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    PYRIEMANN_AVAILABLE = True
except Exception:
    PYRIEMANN_AVAILABLE = False
    print("PyRiemann not found. Riemannian models will be skipped.")

warnings.filterwarnings("ignore")

# ---------- USER CONFIG (edit these) ----------
CSV_PATH = Path("C:/Users/User/Jupyter projects/eeg-analysis/cog_bci/features/features_final_prepared.csv")
COV_MEMMAP_PATH = Path("C:/Users/User/Jupyter projects/eeg-analysis/cog_bci/features/mdm_covs_memmap/covs_float32.dat")
COV_INDEX_MAP = Path("C:/Users/User/Jupyter projects/eeg-analysis/cog_bci/features/mdm_covs_memmap/cov_index_map.csv")
OUT_DIR_BASE = Path("C:/Users/User/Jupyter projects/eeg-analysis/cog_bci/features/loso_results")

# Choose: 'label_3class' or 'label_binary'
LABEL_COL = "label_binary"
SUBJECT_COL = "subject_num"
TIMESTAMPED_OUTPUT = True
RANDOM_STATE = 42

# Parallelism
N_JOBS = 2

# Inner CV / halving
INNER_CV_SPLITS = 3
HALVING_FACTOR = 3
DESIRED_ITERS = 4
MIN_RESOURCES_FLOOR = 200
VERBOSE = 1

# Toggle models
RUN_XGB = True
RUN_RF = True
RUN_MDM = True and PYRIEMANN_AVAILABLE
RUN_TANGENT = True and PYRIEMANN_AVAILABLE

# Base hyperparameter grids (will clamp fs__k dynamically)
XGB_PARAM_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [6, 10],
    "clf__learning_rate": [0.05, 0.1],
    "fs__k": [45, 60]
}
RF_PARAM_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [8, 16],
    "clf__min_samples_leaf": [1, 2],
    "fs__k": [30, 45]
}
TANGENT_PARAM_GRID = {"clf__C": [0.01, 0.1, 1.0]}


OUT_DIR = OUT_DIR_BASE / LABEL_COL
if TIMESTAMPED_OUTPUT:
    OUT_DIR = OUT_DIR / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- helper functions -----------------

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_json_safe(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def clamp_k_list(k_list, n_features):
    vals = set()
    for k in k_list:
        try:
            ki = int(k)
            if ki <= 0:
                continue
            vals.add(min(n_features, ki))
        except Exception:
            continue
    out = sorted(vals)
    if not out:
        # fallback to selecting sqrt of features as conservative choice
        out = [max(1, int(np.sqrt(n_features)))]
    return out


def compute_n_possible_iterations(max_resources, min_resources, factor):
    if min_resources <= 0:
        return 0
    try:
        ratio = float(max_resources) / float(min_resources)
        if ratio < 1:
            return 1
        return 1 + int(floor(log(ratio, factor)))
    except Exception:
        return 0


def align_covs_with_df(cov_index_df, df):
    """
    Attempt to ensure cov memmap rows align with df rows. cov_index_df is read from COV_INDEX_MAP
    Expected columns (one of): 'df_row', 'orig_index', or columns to join on such as ['subject','session','epoch_idx','source_file']
    Returns a reorder index (array of positions) for cov memmap that matches df order, or None if alignment cannot be determined.
    """
    # direct df_row
    if 'df_row' in cov_index_df.columns:
        if (cov_index_df['df_row'].dtype.kind in 'iu'):
            # if cov_index rows are in same order as df, then cov_index_df['df_row'] == np.arange(len(df))
            if np.array_equal(cov_index_df['df_row'].values, np.arange(len(df))):
                return np.arange(len(df))
            else:
                # create mapping from df_row -> position
                mapping = {int(v): i for i, v in enumerate(cov_index_df['df_row'].values)}
                reorder = np.array([mapping.get(i, -1) for i in range(len(df))])
                if (reorder >= 0).all():
                    return reorder
    # try to join by common columns
    join_cols = [c for c in ['subject','session','epoch_idx','source_file'] if c in cov_index_df.columns and c in df.columns]
    if join_cols:
        # build a merge key
        cov_index_df['_join_key'] = cov_index_df[join_cols].astype(str).agg('_'.join, axis=1)
        df['_join_key'] = df[join_cols].astype(str).agg('_'.join, axis=1)
        # try to map df order to cov rows
        cov_map = {k: i for i, k in enumerate(cov_index_df['_join_key'].values)}
        reorder = np.array([cov_map.get(k, -1) for k in df['_join_key'].values])
        df.drop(columns=['_join_key'], inplace=True)
        cov_index_df.drop(columns=['_join_key'], inplace=True)
        if (reorder >= 0).all():
            return reorder
    # alignment unknown
    return None


def aggregate_epoch_preds_to_task(test_idx, y_prob, y_pred, labels_all, df, tid_col='source_file'):
    """
    Aggregate epoch-level predictions to the task/file level.
    Returns: y_task_true, y_task_pred, task_summary dict (tid -> {true, pred, prob, n_epochs})
    """
    test_tasks = df.loc[test_idx, tid_col].values
    task_probs = defaultdict(list)
    task_true = {}
    for i, idx in enumerate(test_idx):
        tid = test_tasks[i]
        task_true[tid] = int(labels_all[idx])
        if y_prob is not None:
            task_probs[tid].append(y_prob[i])
        else:
            # fallback to one-hot vector
            n_classes = int(labels_all.max() + 1)
            vec = np.zeros(n_classes, dtype=float)
            vec[int(y_pred[i])] = 1.0
            task_probs[tid].append(vec)
    mean_probs = {tid: np.mean(v, axis=0) for tid, v in task_probs.items()}
    task_pred = {tid: int(np.argmax(p)) for tid, p in mean_probs.items()}
    y_task_true = np.array([task_true[tid] for tid in task_pred.keys()])
    y_task_pred = np.array([task_pred[tid] for tid in task_pred.keys()])

    # prepare a serializable summary for each task
    task_summary = {}
    for tid, p in mean_probs.items():
        task_summary[tid] = {
            'true': int(task_true[tid]),
            'pred': int(task_pred[tid]),
            'prob': make_json_safe(p),
            'n_epochs': int(len(task_probs[tid]))
        }
    return y_task_true, y_task_pred, task_summary


# ----------------- Data Loading & initial checks -----------------
print(f"[{now()}] Loading features CSV...")
df = pd.read_csv(CSV_PATH)
df['_orig_row'] = np.arange(len(df))

# # after df = pd.read_csv(...)
# print("All columns:", df.columns.tolist())
#
# # columns you already treat as non-features
# id_cols = {"subject","subject_num","session","session_num","task","epoch_idx","source_file","pc_dist","is_outlier"}
# excluded_labels = {"label_3class","label_binary","rsme_score"}
#
# # candidate unexpected helpers that often slip in
# helpers = [c for c in ['_orig_row','_join_key','df_row','_tmp'] if c in df.columns]
# print("Helper columns present:", helpers)
#
# # compute feat_cols the same way your script does, then show the extra one
# feat_cols = [c for c in df.columns if c not in id_cols and c not in excluded_labels]
# print("n features (counted):", len(feat_cols))
# print("Extra columns counted as features (candidates):", [c for c in feat_cols if c in helpers])
# # also print any small column names that look like helpers
# print("Small/leading-underscore columns in feat_cols:", [c for c in feat_cols if c.startswith('_')])

id_cols = {"subject", "subject_num", "session", "session_num", "task", "epoch_idx", "source_file", "pc_dist", "is_outlier"}
feat_cols = [c for c in df.columns if c not in id_cols and c not in ("label_3class","label_binary","rsme_score")]
print(f"[{now()}] Total rows: {df.shape[0]}, features (tabular) count: {len(feat_cols)}")

cov_index = None
covs = None
reorder = None
if RUN_MDM or RUN_TANGENT:
    cov_index = pd.read_csv(COV_INDEX_MAP)
    reorder = align_covs_with_df(cov_index, df)  # computed on original df with _orig_row present
    if reorder is None:
        print(f"[{now()}] WARNING: Could not automatically align cov_index rows to df rows. Proceeding but please verify cov_index.")
    memmap_temp = np.memmap(COV_MEMMAP_PATH, dtype=np.float32, mode='r')
    n_rows = cov_index.shape[0]
    n_ch = int(round((memmap_temp.size / n_rows) ** 0.5))
    covs = np.memmap(COV_MEMMAP_PATH, dtype=np.float32, mode='r', shape=(n_rows, n_ch, n_ch))
    print(f"[{now()}] Cov memmap loaded: rows={n_rows}, channels={n_ch}")
else:
    print(f"[{now()}] Riemannian models disabled; skipping cov_index/covs load.")

# # load cov_index and memmap metadata
# cov_index = pd.read_csv(COV_INDEX_MAP)
# # attempt to align cov rows to df rows
# reorder = align_covs_with_df(cov_index, df)
# if reorder is None:
#     print(f"[{now()}] WARNING: Could not automatically align cov_index rows to df rows. Assuming the memmap rows are in the same order as df.")

# # load memmap (we'll index into it using reorder if found)
# memmap_temp = np.memmap(COV_MEMMAP_PATH, dtype=np.float32, mode='r')
# # If cov_index contains number of rows, try to infer n_ch
# n_rows = cov_index.shape[0]
# n_ch = int(round((memmap_temp.size / n_rows) ** 0.5))
# covs = np.memmap(COV_MEMMAP_PATH, dtype=np.float32, mode='r', shape=(n_rows, n_ch, n_ch))
# print(f"[{now()}] Cov memmap loaded: rows={n_rows}, channels={n_ch}")

# Prepare labels & subjects (handle binary filtering if requested)
labels_raw = pd.to_numeric(df[LABEL_COL], errors='coerce')

# if LABEL_COL == 'label_binary':
#     valid_mask = labels_raw.notna() & np.isfinite(labels_raw)
#     n_dropped = int((~valid_mask).sum())
#     if n_dropped > 0:
#         print(f"[{now()}] NOTE: Detected {n_dropped} rows with missing/invalid '{LABEL_COL}'. These will be excluded for the binary run.")
#     selected_idx = np.where(valid_mask)[0]
#     df = df.loc[valid_mask].reset_index(drop=True)
#     cov_index = cov_index.loc[valid_mask].reset_index(drop=True)
#     # if there was a reorder decision earlier, we must apply it before subsetting; here we assume original memmap order == original df
#     covs = covs[selected_idx].astype(np.float64)
#     labels = labels_raw[valid_mask].round().astype(int).values
#     subjects = df[SUBJECT_COL].values
# else:
#     if labels_raw.isna().any():
#         n_bad = int(labels_raw.isna().sum())
#         raise ValueError(f"Found {n_bad} missing values in {LABEL_COL} when running. Please check CSV.")
#     labels = labels_raw.round().astype(int).values
#     subjects = df[SUBJECT_COL].values


if LABEL_COL == 'label_binary':
    # labels_raw = pd.to_numeric(df[LABEL_COL], errors='coerce')
    valid_mask = labels_raw.notna() & np.isfinite(labels_raw)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"[{now()}] NOTE: Detected {n_dropped} rows with missing/invalid '{LABEL_COL}'. These will be excluded for the binary run.")

    # original row positions to keep (safe)
    selected_orig_rows = df.loc[valid_mask, '_orig_row'].values.astype(int)

    # subset df (and reset index for model training)
    df = df.loc[valid_mask].reset_index(drop=True)

    # if covs are present, subset them by original-row indices (unambiguous)
    if covs is not None and cov_index is not None:
        # preferred: if cov_index contains 'df_row' use that
        if 'df_row' in cov_index.columns:
            mask_cov = cov_index['df_row'].isin(selected_orig_rows).values
            cov_index = cov_index.loc[mask_cov].reset_index(drop=True)
            selected_idx = np.where(mask_cov)[0]
            covs = covs[selected_idx].astype(np.float64)
        else:
            # fallback: assume memmap rows were written in original CSV order
            if selected_orig_rows.max() >= covs.shape[0]:
                raise ValueError("Selected original row index exceeds cov memmap rows — check cov_index creation.")
            covs = covs[selected_orig_rows].astype(np.float64)
            cov_index = cov_index.loc[selected_orig_rows].reset_index(drop=True)
    labels = labels_raw[valid_mask].round().astype(int).values
    subjects = df[SUBJECT_COL].values

# if reorder mapping exists, reorder covs to match df order (after potential filtering above)
if reorder is not None:
    # reorder may refer to original df length before filtering; attempt to adapt
    if len(reorder) == len(df):
        covs = covs[reorder]
    else:
        # try to map using cov_index join columns again
        print(f"[{now()}] NOTE: reorder length ({len(reorder)}) != df length ({len(df)}). Skipping automatic reorder; ensure memmap alignment manually.")

n_rows = covs.shape[0]

for temp_col in ['_orig_row', '_join_key']:
    if temp_col in df.columns:
        df.drop(columns=[temp_col], inplace=True)

feat_cols = [c for c in df.columns if c not in id_cols and c not in ("label_3class","label_binary","rsme_score")]
unique_subjects = np.unique(subjects)
print(f"[{now()}] Data Ready. Rows: {n_rows}, Features: {len(feat_cols)}, Subjects: {len(unique_subjects)}")
print(f"[{now()}] Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

assert len(df) == covs.shape[0] == len(subjects) == len(labels), "Alignment error: df/covs/labels/subjects lengths differ!"
if covs is not None:
    # also check cov_index if available
    if cov_index is not None:
        assert len(df) == covs.shape[0] == cov_index.shape[0], "Alignment error: df/covs/cov_index lengths differ!"
    else:
        assert len(df) == covs.shape[0], "Alignment error: df/covs lengths differ!"

# clamp feature-selection k values based on n_features
n_features = len(feat_cols)
XGB_PARAM_GRID['fs__k'] = clamp_k_list(XGB_PARAM_GRID.get('fs__k', [max(1, int(np.sqrt(n_features)))]), n_features)
RF_PARAM_GRID['fs__k'] = clamp_k_list(RF_PARAM_GRID.get('fs__k', [max(1, int(np.sqrt(n_features)))]), n_features)

print(f"[{now()}] Clamped fs__k: XGB {XGB_PARAM_GRID['fs__k']}, RF {RF_PARAM_GRID['fs__k']}")

# container for fold summaries
all_fold_results = []

# ----------------- LOSO Outer Loop -----------------
for test_subject in unique_subjects:
    subj_id = int(test_subject)
    summary_path = OUT_DIR / f"fold_{subj_id}_summary.json"

    if summary_path.exists():
        print(f"[{now()}] Fold {subj_id}: Checkpoint found. Skipping.")
        continue

    print(f"[{now()}] Starting Fold {subj_id} (Test Subject: {subj_id})")

    test_mask = (subjects == test_subject)
    train_mask = ~test_mask
    train_idx, test_idx = np.where(train_mask)[0], np.where(test_mask)[0]

    X_train_tab = df.loc[train_idx, feat_cols].values
    X_test_tab = df.loc[test_idx, feat_cols].values
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = subjects[train_idx].astype(int)

    inner_cv = GroupKFold(n_splits=min(INNER_CV_SPLITS, len(np.unique(groups_train))))
    fold_summary = {
        "test_subject": int(test_subject),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "train_label_counts": {int(k): int(v) for k, v in Counter(y_train).items()},
        "test_label_counts": {int(k): int(v) for k, v in Counter(y_test).items()},
        "n_tasks_test": int(len(np.unique(df.loc[test_idx, 'source_file']))),
        "models": {}
    }

    # compute dynamic min_resources per fold
    n_train = len(train_idx)
    denom = HALVING_FACTOR ** (DESIRED_ITERS - 1)
    if denom <= 0:
        denom = 1
    min_resources_dynamic = max(MIN_RESOURCES_FLOOR, n_train // denom)
    min_resources_dynamic = min(min_resources_dynamic, n_train)
    n_possible_iters = compute_n_possible_iterations(n_train, min_resources_dynamic, HALVING_FACTOR)
    print(f"[{now()}] Fold {subj_id}: n_train={n_train}, min_resources={min_resources_dynamic}, n_possible_iterations={n_possible_iters}")

    # ----- model loop utility -----
    def safe_search_fit(search_obj, Xtr, ytr, groups_tr, model_name):
        """Fit search_obj only if there are >=2 classes in ytr and inner_cv can be constructed."""
        if len(np.unique(ytr)) < 2:
            msg = f"Training skipped for {model_name}: less than 2 classes in training set"
            print(f"[{now()}] {msg}")
            return {'error': msg}
        try:
            search_obj.fit(Xtr, ytr, groups=groups_tr)
            return {'search': search_obj}
        except Exception as e:
            return {'error': str(e)}

    # ---------------- XGB ----------------
    if RUN_XGB:
        print(f"[{now()}]  - Running XGB Halving Search...")
        pipe_xgb = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("fs", SelectKBest(score_func=f_classif, k=min(45, n_features))),
            ("clf", XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"))
        ])
        search = HalvingGridSearchCV(estimator=pipe_xgb,
                                     param_grid=XGB_PARAM_GRID,
                                     cv=inner_cv,
                                     factor=HALVING_FACTOR,
                                     min_resources=min_resources_dynamic,
                                     scoring="balanced_accuracy",
                                     n_jobs=N_JOBS,
                                     verbose=VERBOSE,
                                     error_score=np.nan,
                                     return_train_score=False)
        result = safe_search_fit(search, X_train_tab, y_train, groups_train, 'XGB')
        if 'error' in result:
            fold_summary['models']['XGB'] = {'error': result['error']}
        else:
            search = result['search']
            best = search.best_estimator_
            print(f"[{now()}] XGB best inner score: {search.best_score_:.4f} best_params: {search.best_params_}")
            y_pred = best.predict(X_test_tab)
            y_prob = best.predict_proba(X_test_tab) if hasattr(best, 'predict_proba') else None

            # epoch-level metrics (diagnostic)
            balacc_epoch = balanced_accuracy_score(y_test, y_pred)
            acc_epoch = accuracy_score(y_test, y_pred)
            f1_epoch = f1_score(y_test, y_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
            cm_epoch = confusion_matrix(y_test, y_pred)

            # AUC (epoch-level) when probabilities exist
            auc_epoch = None
            if y_prob is not None:
                try:
                    if LABEL_COL == 'label_binary':
                        auc_epoch = float(roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        auc_epoch = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))
                except Exception:
                    auc_epoch = None

            # aggregate to task-level
            y_task_true, y_task_pred, task_summary = aggregate_epoch_preds_to_task(test_idx, y_prob, y_pred, labels, df, tid_col='source_file')
            balacc_task = balanced_accuracy_score(y_task_true, y_task_pred)
            acc_task = accuracy_score(y_task_true, y_task_pred)
            f1_task = f1_score(y_task_true, y_task_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
            cm_task = confusion_matrix(y_task_true, y_task_pred)

            # AUC (task-level) when probabilities exist
            auc_task = None
            if y_prob is not None:
                try:
                    # build matrix of mean probs in the same order as y_task_true
                    mean_probs = {tid: np.array(task_summary[tid]['prob']) for tid in task_summary.keys()}
                    probs_arr = np.vstack([mean_probs[tid] for tid in task_summary.keys()])
                    if LABEL_COL == 'label_binary':
                        auc_task = float(roc_auc_score(y_task_true, probs_arr[:, 1]))
                    else:
                        auc_task = float(roc_auc_score(y_task_true, probs_arr, multi_class='ovr', average='macro'))
                except Exception:
                    auc_task = None

            fold_summary['models']['XGB'] = {
                'inner_best_score': float(search.best_score_),
                'inner_best_params': make_json_safe(search.best_params_),
                'test_balanced_accuracy_epoch': float(balacc_epoch),
                'test_accuracy_epoch': float(acc_epoch),
                'test_f1_epoch': float(f1_epoch),
                'confusion_matrix_epoch': cm_epoch.tolist(),
                'test_auc_epoch': (float(auc_epoch) if auc_epoch is not None else None),
                'test_balanced_accuracy_task': float(balacc_task),
                'test_accuracy_task': float(acc_task),
                'test_f1_task': float(f1_task),
                'confusion_matrix_task': cm_task.tolist(),
                'test_auc_task': (float(auc_task) if auc_task is not None else None),
                'n_tasks_test': int(len(task_summary)),
                'task_level_summary': make_json_safe(task_summary)
            }

            # save objects
            joblib.dump(search, OUT_DIR / f"fold_{subj_id}_XGB_halving.joblib")
            joblib.dump(best, OUT_DIR / f"fold_{subj_id}_XGB_best.joblib")

    # ---------------- RF ----------------
    if RUN_RF:
        print(f"[{now()}]  - Running RF Halving Search...")
        pipe_rf = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("fs", SelectKBest(score_func=f_classif, k=min(30, n_features))),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced'))
        ])
        search = HalvingGridSearchCV(estimator=pipe_rf,
                                     param_grid=RF_PARAM_GRID,
                                     cv=inner_cv,
                                     factor=HALVING_FACTOR,
                                     min_resources=min_resources_dynamic,
                                     scoring="balanced_accuracy",
                                     n_jobs=N_JOBS,
                                     verbose=VERBOSE,
                                     error_score=np.nan,
                                     return_train_score=False)
        result = safe_search_fit(search, X_train_tab, y_train, groups_train, 'RF')
        if 'error' in result:
            fold_summary['models']['RF'] = {'error': result['error']}
        else:
            search = result['search']
            best = search.best_estimator_
            print(f"[{now()}] RF best inner score: {search.best_score_:.4f} best_params: {search.best_params_}")
            y_pred = best.predict(X_test_tab)
            y_prob = best.predict_proba(X_test_tab) if hasattr(best, 'predict_proba') else None

            balacc_epoch = balanced_accuracy_score(y_test, y_pred)
            acc_epoch = accuracy_score(y_test, y_pred)
            f1_epoch = f1_score(y_test, y_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
            cm_epoch = confusion_matrix(y_test, y_pred)

            auc_epoch = None
            if y_prob is not None:
                try:
                    if LABEL_COL == 'label_binary':
                        auc_epoch = float(roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        auc_epoch = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))
                except Exception:
                    auc_epoch = None

            y_task_true, y_task_pred, task_summary = aggregate_epoch_preds_to_task(test_idx, y_prob, y_pred, labels, df, tid_col='source_file')
            balacc_task = balanced_accuracy_score(y_task_true, y_task_pred)
            acc_task = accuracy_score(y_task_true, y_task_pred)
            f1_task = f1_score(y_task_true, y_task_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
            cm_task = confusion_matrix(y_task_true, y_task_pred)

            auc_task = None
            if y_prob is not None:
                try:
                    mean_probs = {tid: np.array(task_summary[tid]['prob']) for tid in task_summary.keys()}
                    probs_arr = np.vstack([mean_probs[tid] for tid in task_summary.keys()])
                    if LABEL_COL == 'label_binary':
                        auc_task = float(roc_auc_score(y_task_true, probs_arr[:, 1]))
                    else:
                        auc_task = float(roc_auc_score(y_task_true, probs_arr, multi_class='ovr', average='macro'))
                except Exception:
                    auc_task = None

            fold_summary['models']['RF'] = {
                'inner_best_score': float(search.best_score_),
                'inner_best_params': make_json_safe(search.best_params_),
                'test_balanced_accuracy_epoch': float(balacc_epoch),
                'test_accuracy_epoch': float(acc_epoch),
                'test_f1_epoch': float(f1_epoch),
                'confusion_matrix_epoch': cm_epoch.tolist(),
                'test_auc_epoch': (float(auc_epoch) if auc_epoch is not None else None),
                'test_balanced_accuracy_task': float(balacc_task),
                'test_accuracy_task': float(acc_task),
                'test_f1_task': float(f1_task),
                'confusion_matrix_task': cm_task.tolist(),
                'test_auc_task': (float(auc_task) if auc_task is not None else None),
                'n_tasks_test': int(len(task_summary)),
                'task_level_summary': make_json_safe(task_summary)
            }

            joblib.dump(search, OUT_DIR / f"fold_{subj_id}_RF_halving.joblib")
            joblib.dump(best, OUT_DIR / f"fold_{subj_id}_RF_best.joblib")

    # ---------------- MDM (Riemann) ----------------
    if RUN_MDM:
        try:
            print(f"[{now()}]  - Running MDM (Riemann)...")
            covs_train = covs[train_idx].astype(np.float64)
            covs_test = covs[test_idx].astype(np.float64)

            mdm = MDM()
            if len(np.unique(y_train)) < 2:
                raise ValueError("MDM: training set has <2 classes; skipping")
            mdm.fit(covs_train, y_train)
            y_pred = mdm.predict(covs_test)
            y_prob = None
            balacc_epoch = balanced_accuracy_score(y_test, y_pred)
            acc_epoch = accuracy_score(y_test, y_pred)
            f1_epoch = f1_score(y_test, y_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
            cm_epoch = confusion_matrix(y_test, y_pred)

            y_task_true, y_task_pred, task_summary = aggregate_epoch_preds_to_task(test_idx, y_prob, y_pred, labels, df, tid_col='source_file')
            balacc_task = balanced_accuracy_score(y_task_true, y_task_pred)
            acc_task = accuracy_score(y_task_true, y_task_pred)
            f1_task = f1_score(y_task_true, y_task_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
            cm_task = confusion_matrix(y_task_true, y_task_pred)

            fold_summary['models']['MDM'] = {
                'test_balanced_accuracy_epoch': float(balacc_epoch),
                'test_accuracy_epoch': float(acc_epoch),
                'test_f1_epoch': float(f1_epoch),
                'confusion_matrix_epoch': cm_epoch.tolist(),
                'test_balanced_accuracy_task': float(balacc_task),
                'test_accuracy_task': float(acc_task),
                'test_f1_task': float(f1_task),
                'confusion_matrix_task': cm_task.tolist(),
                'test_auc_epoch': None,
                'test_auc_task': None,
                'n_tasks_test': int(len(task_summary)),
                'task_level_summary': make_json_safe(task_summary)
            }
            joblib.dump(mdm, OUT_DIR / f"fold_{subj_id}_MDM.joblib")
        except Exception as e:
            print(f"[{now()}] ERROR in MDM for fold {subj_id}: {e}")
            fold_summary['models']['MDM'] = {'error': str(e)}

    # ---------------- TangentSpace + LR (Riemann) ----------------
    if RUN_TANGENT:
        try:
            print(f"[{now()}]  - Running TangentLR Search...")
            covs_train = covs[train_idx].astype(np.float64)
            covs_test = covs[test_idx].astype(np.float64)
            ts = TangentSpace()
            X_train_ts, X_test_ts = ts.fit_transform(covs_train), ts.transform(covs_test)

            pipe_ts_lr = Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", random_state=RANDOM_STATE))
            ])
            search = HalvingGridSearchCV(estimator=pipe_ts_lr,
                                         param_grid=TANGENT_PARAM_GRID,
                                         cv=inner_cv,
                                         factor=HALVING_FACTOR,
                                         min_resources=min_resources_dynamic,
                                         scoring="balanced_accuracy",
                                         n_jobs=N_JOBS,
                                         verbose=VERBOSE,
                                         error_score=np.nan,
                                         return_train_score=False)
            result = safe_search_fit(search, X_train_ts, y_train, groups_train, 'TangentLR')
            if 'error' in result:
                fold_summary['models']['TangentLR'] = {'error': result['error']}
            else:
                search = result['search']
                best = search.best_estimator_
                y_pred = best.predict(X_test_ts)
                y_prob = best.predict_proba(X_test_ts) if hasattr(best, 'predict_proba') else None

                balacc_epoch = balanced_accuracy_score(y_test, y_pred)
                acc_epoch = accuracy_score(y_test, y_pred)
                f1_epoch = f1_score(y_test, y_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
                cm_epoch = confusion_matrix(y_test, y_pred)

                auc_epoch = None
                if y_prob is not None:
                    try:
                        if LABEL_COL == 'label_binary':
                            auc_epoch = float(roc_auc_score(y_test, y_prob[:, 1]))
                        else:
                            auc_epoch = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))
                    except Exception:
                        auc_epoch = None

                y_task_true, y_task_pred, task_summary = aggregate_epoch_preds_to_task(test_idx, y_prob, y_pred, labels, df, tid_col='source_file')
                balacc_task = balanced_accuracy_score(y_task_true, y_task_pred)
                acc_task = accuracy_score(y_task_true, y_task_pred)
                f1_task = f1_score(y_task_true, y_task_pred, average=('binary' if LABEL_COL=='label_binary' else 'macro'))
                cm_task = confusion_matrix(y_task_true, y_task_pred)

                auc_task = None
                if y_prob is not None:
                    try:
                        mean_probs = {tid: np.array(task_summary[tid]['prob']) for tid in task_summary.keys()}
                        probs_arr = np.vstack([mean_probs[tid] for tid in task_summary.keys()])
                        if LABEL_COL == 'label_binary':
                            auc_task = float(roc_auc_score(y_task_true, probs_arr[:, 1]))
                        else:
                            auc_task = float(roc_auc_score(y_task_true, probs_arr, multi_class='ovr', average='macro'))
                    except Exception:
                        auc_task = None

                fold_summary['models']['TangentLR'] = {
                    'inner_best_score': float(search.best_score_),
                    'inner_best_params': make_json_safe(search.best_params_),
                    'test_balanced_accuracy_epoch': float(balacc_epoch),
                    'test_accuracy_epoch': float(acc_epoch),
                    'test_f1_epoch': float(f1_epoch),
                    'confusion_matrix_epoch': cm_epoch.tolist(),
                    'test_auc_epoch': (float(auc_epoch) if auc_epoch is not None else None),
                    'test_balanced_accuracy_task': float(balacc_task),
                    'test_accuracy_task': float(acc_task),
                    'test_f1_task': float(f1_task),
                    'confusion_matrix_task': cm_task.tolist(),
                    'test_auc_task': (float(auc_task) if auc_task is not None else None),
                    'n_tasks_test': int(len(task_summary)),
                    'task_level_summary': make_json_safe(task_summary)
                }

                joblib.dump(search, OUT_DIR / f"fold_{subj_id}_TangentLR_halving.joblib")
                joblib.dump(best, OUT_DIR / f"fold_{subj_id}_TangentLR_best.joblib")
        except Exception as e:
            print(f"[{now()}] ERROR in TangentLR for fold {subj_id}: {e}")
            fold_summary['models']['TangentLR'] = {'error': str(e)}

    # Save fold_summary and append
    all_fold_results.append(fold_summary)
    save_json(OUT_DIR / f"fold_{int(test_subject)}_summary.json", make_json_safe(fold_summary))

    # Print brief test results summary (safe access)
    model_balaccs = {}
    for m, info in fold_summary["models"].items():
        if isinstance(info, dict) and "test_balanced_accuracy_task" in info:
            model_balaccs[m] = info["test_balanced_accuracy_task"]
        else:
            model_balaccs[m] = None
    print(f"[{now()}] Fold {int(test_subject)} done. Test balanced accuracies (task-level): {model_balaccs}")

    # Memory Management: Clean up large objects
    del X_train_tab, X_test_tab, y_train, y_test
    if RUN_MDM or RUN_TANGENT:
        try:
            del covs_train, covs_test
        except Exception:
            pass
    gc.collect()

# Save overall summary
master_summary_path = OUT_DIR / (f"all_folds_summary_{now().replace(' ', '_').replace(':','-')}.json" if TIMESTAMPED_OUTPUT else "all_folds_summary.json")
save_json(master_summary_path, make_json_safe(all_fold_results))
print(f"[{now()}] ALL FOLDS COMPLETED. Summary JSONs are in {OUT_DIR}")