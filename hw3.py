# streamlit_app.py
# ==========================================================
# Spam Classifier Dashboard (Streamlit)
# - Data: https://github.com/benjamine9721025/o/tree/main/datasets/sms_spam_no_header.csv
# - Model: Logistic Regression
# - Deterministic & idempotent text cleaning pipeline
# ==========================================================
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, roc_curve, auc,
    precision_recall_curve
)

st.set_page_config(page_title="Spam Classifier Dashboard", layout="wide")

# -------------------------------
# Constants & Utilities
# -------------------------------
RAW_URL = "https://raw.githubusercontent.com/benjamine9721025/o/main/datasets/sms_spam_no_header.csv"

LABEL_POS = "spam"
LABEL_NEG = "ham"

RANDOM_STATE = 42

# Regex patterns (compiled once)
URL_RE   = re.compile(r"""(?i)\b((?:https?://|www\.)\S+)\b""")
EMAIL_RE = re.compile(r"""(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b""")
PHONE_RE = re.compile(r"""(?i)\b(?:\+?\d[\d\-\s]{7,}\d)\b""")
NUM_RE   = re.compile(r"""\b\d+(?:[.,]\d+)?\b""")

# Keep letters, digits, spaces, and angle-bracket tokens like <URL>, <EMAIL>...
#   - allow <, > explicitly; remove other punctuation
PUNCT_RE = re.compile(r"""[^A-Za-z0-9<>\s]""")

STOP_SET = set(ENGLISH_STOP_WORDS)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def text_clean(s: str, remove_stopwords: bool=False, mask_numbers: bool=True) -> str:
    """
    Deterministic, idempotent cleaning:
      1) lower + trim spaces
      2) mask URL/EMAIL/PHONE/(optional) NUM
      3) remove punctuation (keep word chars, spaces, and <URL>/<EMAIL>/...)
      4) optional stopword removal (default OFF in UI)
    """
    if s is None:
        return ""
    x = s.lower().strip()

    # step 2: mask special patterns
    x = URL_RE.sub("<URL>", x)
    x = EMAIL_RE.sub("<EMAIL>", x)
    x = PHONE_RE.sub("<PHONE>", x)
    if mask_numbers:
        x = NUM_RE.sub("<NUM>", x)

    # step 3: remove punctuation (except angle brackets)
    x = PUNCT_RE.sub(" ", x)

    # canonicalize spaces
    x = normalize_spaces(x)

    # step 4: optional stopword removal
    if remove_stopwords:
        toks = x.split(" ")
        toks = [t for t in toks if t and t not in STOP_SET]
        x = " ".join(toks)

    return x

# A tokenizer that treats cleaned text as pre-tokenized space-separated tokens
def identity_tokenizer(tokens_str: str):
    # TfidfVectorizer expects list of tokens if we give a tokenizer
    return tokens_str.split(" ") if tokens_str else []


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Try GitHub raw first; fallback to local file if user puts it beside the app.
    File expected format: 2 columns without header: [label, text]
    labels: 'ham' or 'spam'
    """
    try:
        df = pd.read_csv(RAW_URL, header=None, names=["label", "text"], encoding="utf-8")
    except Exception:
        # fallback to local path (same filename)
        df = pd.read_csv("sms_spam_no_header.csv", header=None, names=["label", "text"], encoding="utf-8")
    # clean obvious NaN
    df["text"] = df["text"].fillna("")
    # normalize labels
    df["label"] = df["label"].str.strip().str.lower().map({"spam": LABEL_POS, "ham": LABEL_NEG})
    df = df[df["label"].isin({LABEL_POS, LABEL_NEG})].reset_index(drop=True)
    return df


def fit_vectorizer_and_model(df: pd.DataFrame, remove_stopwords: bool, mask_numbers: bool):
    # Clean deterministically
    cleaned = df["text"].apply(lambda s: text_clean(s, remove_stopwords=remove_stopwords, mask_numbers=mask_numbers))

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    # Vectorizer: use identity preprocessor + tokenizer to preserve <URL> tokens etc.
    vect = TfidfVectorizer(
        preprocessor=lambda s: s,           # already cleaned
        tokenizer=identity_tokenizer,       # split by space
        token_pattern=None,                 # must set None when using custom tokenizer
        ngram_range=(1,2),                  # unigrams + bigrams work well for spam
        min_df=2
    )
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    # Logistic Regression (deterministic)
    clf = LogisticRegression(
        solver="liblinear", C=1.0, random_state=RANDOM_STATE, max_iter=200
    )
    clf.fit(Xtr, y_train)

    return vect, clf, (X_train, X_test, y_train, y_test)


def compute_top_tokens_per_class(cleaned_series: pd.Series, y: pd.Series, topk=20):
    dfc = pd.DataFrame({"x": cleaned_series, "y": y}).dropna()
    top = {}
    for cls in [LABEL_NEG, LABEL_POS]:
        toks = []
        for s in dfc.loc[dfc["y"]==cls, "x"]:
            toks.extend([t for t in s.split(" ") if t])
        ctr = Counter(toks)
        top[cls] = ctr.most_common(topk)
    return top


def plot_barh(items, title):
    if not items:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center")
        st.pyplot(fig, use_container_width=True)
        return
    words, counts = zip(*items)
    idx = np.arange(len(words))
    fig, ax = plt.subplots()
    ax.barh(idx, counts)
    ax.set_yticks(idx, labels=words)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Count")
    st.pyplot(fig, use_container_width=True)


def plot_confusion(cm, labels=(LABEL_NEG, LABEL_POS), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    st.pyplot(fig, use_container_width=True)


def plot_roc(y_true_bin, scores, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    st.pyplot(fig, use_container_width=True)


def plot_pr(y_true_bin, scores, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true_bin, scores)
    ap = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    st.pyplot(fig, use_container_width=True)


# -------------------------------
# Sidebar: Controls
# -------------------------------
st.sidebar.header("Preprocessing & Options")
opt_remove_stop = st.sidebar.checkbox("Remove stopwords", value=False)
opt_mask_numbers = st.sidebar.checkbox("Mask numbers as <NUM>", value=True)
topk = st.sidebar.slider("Top-N keywords per class", 5, 40, 20, 1)
threshold = st.sidebar.slider("Decision threshold (predict spam if P(spam) ≥ τ)", 0.05, 0.95, 0.50, 0.01)

st.sidebar.caption("⚙️ 以上選項會影響模型或展示。")

# -------------------------------
# Data Loading
# -------------------------------
st.title("📨 Spam Classifier — Interactive Dashboard (Logistic Regression)")
with st.expander("資料來源與欄位說明", expanded=True):
    st.markdown(
        f"- 來源檔案：`sms_spam_no_header.csv`（GitHub Raw 讀取）\n"
        f"- 欄位：**無標題** → 第一欄 `label`（`ham`/`spam`），第二欄 `text`（訊息內容）\n"
        f"- 載入優先序：GitHub Raw → 本地同名檔案。"
    )

df = load_data()
st.success(f"✅ 已載入資料，共 {len(df)} 筆。")

# -------------------------------
# Cleaning + Train/Test split + Fit
# -------------------------------
cleaned_all = df["text"].apply(lambda s: text_clean(s, remove_stopwords=opt_remove_stop, mask_numbers=opt_mask_numbers))
vect, clf, (X_train, X_test, y_train, y_test) = fit_vectorizer_and_model(df, opt_remove_stop, opt_mask_numbers)

# Also keep cleaned versions for analysis
train_mask = df.index.isin(y_train.index)  # y_train is a Series view; safer to recompute
# 重新用相同 random_state 分割，以取得對應 cleaned 的 train/test
Xtr_all, Xte_all, ytr_all, yte_all = train_test_split(
    cleaned_all, df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
)

# -------------------------------
# (4) Data Analysis: class ratio & token stats
# -------------------------------
st.subheader("資料分析 / 分類比例")
col1, col2 = st.columns(2)
with col1:
    counts = df["label"].value_counts().reindex([LABEL_NEG, LABEL_POS]).fillna(0).astype(int)
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.dataframe(
        pd.DataFrame({"label": counts.index, "count": counts.values, "ratio": (counts.values/len(df)).round(3)})
    )

st.subheader("關鍵字統計（每類最常見詞彙）")
tops = compute_top_tokens_per_class(Xtr_all, ytr_all, topk=topk)
c1, c2 = st.columns(2)
with c1:
    plot_barh(tops.get(LABEL_NEG, []), f"Top-{topk} tokens in {LABEL_NEG}")
with c2:
    plot_barh(tops.get(LABEL_POS, []), f"Top-{topk} tokens in {LABEL_POS}")

# -------------------------------
# (6) Metrics & Visualizations
# -------------------------------
st.subheader("模型效能視覺化")

Xtr_vec = vect.transform(Xtr_all)
Xte_vec = vect.transform(Xte_all)

y_pred = clf.predict(Xte_vec)
y_prob = clf.predict_proba(Xte_vec)[:, 1]  # P(spam)

# Confusion Matrix with chosen threshold
y_pred_tau = (y_prob >= threshold).astype(int)
# Map textual labels to 0/1
label_to_bin = {LABEL_NEG: 0, LABEL_POS: 1}
y_true_bin = yte_all.map(label_to_bin).values
cm = confusion_matrix(y_true_bin, y_pred_tau, labels=[0,1])

c1, c2, c3 = st.columns(3)
with c1:
    plot_confusion(cm, labels=(LABEL_NEG, LABEL_POS), title=f"Confusion @ τ={threshold:.2f}")
with c2:
    # ROC uses score/probability, not thresholded label
    plot_roc(y_true_bin, y_prob)
with c3:
    plot_pr(y_true_bin, y_prob)

# F1 report (macro/weighted + per-class)
st.markdown("#### F1 分數表")
f1_macro = f1_score(y_true_bin, y_pred_tau, average="macro", zero_division=0)
f1_weighted = f1_score(y_true_bin, y_pred_tau, average="weighted", zero_division=0)
report = classification_report(
    y_true_bin, y_pred_tau, target_names=[LABEL_NEG, LABEL_POS], output_dict=True, zero_division=0
)
rep_df = pd.DataFrame(report).transpose()
rep_df.loc["macro avg", "f1-score"] = f1_macro
rep_df.loc["weighted avg", "f1-score"] = f1_weighted
st.dataframe(rep_df.round(3), use_container_width=True)

# -------------------------------
# (7) Live Inference
# -------------------------------
st.subheader("即時預測（Live Inference）")

ex_spam = None
ex_ham = None
# Pick simple examples from dataset (first occurence)
try:
    ex_spam = df[df["label"]==LABEL_POS]["text"].iloc[0]
except Exception:
    ex_spam = "Congratulations! You’ve won a free prize. Visit http://scam.example now!!!"
try:
    ex_ham = df[df["label"]==LABEL_NEG]["text"].iloc[0]
except Exception:
    ex_ham = "Hey, are we still meeting for lunch today?"

b1, b2 = st.columns(2)
with b1:
    if st.button("Use spam example"):
        st.session_state["live_text"] = ex_spam
with b2:
    if st.button("Use ham example"):
        st.session_state["live_text"] = ex_ham

default_txt = st.session_state.get("live_text", "")
user_txt = st.text_area(
    "輸入訊息文字（會套用與訓練相同的正規化與遮罩：URL/EMAIL/PHONE/<NUM>）",
    value=default_txt, height=120
)

if st.button("預測"):
    cleaned = text_clean(user_txt, remove_stopwords=opt_remove_stop, mask_numbers=opt_mask_numbers)
    Xlive = vect.transform([cleaned])
    prob_spam = float(clf.predict_proba(Xlive)[:, 1][0])
    pred_label = LABEL_POS if prob_spam >= threshold else LABEL_NEG

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**Normalized text:**")
        st.code(cleaned or "(empty)", language="text")
        st.metric("Predicted Label", pred_label, help=f"τ={threshold:.2f}, predict spam if P(spam)≥τ")
    with c2:
        st.markdown("**Spam probability**")
        st.write(f"P(spam) = **{prob_spam:.3f}**  (threshold τ = {threshold:.2f})")
        # Visual probability bar
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([0], [prob_spam])
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.axvline(threshold, linestyle="--")
        ax.set_xlabel("Probability of spam")
        st.pyplot(fig, use_container_width=True)

# -------------------------------
# (2) Preprocessing demo (deterministic & idempotent)
# -------------------------------
with st.expander("顯示部分清理後樣本（驗證確定性/冪等）"):
    sample_df = df.sample(n=min(8, len(df)), random_state=RANDOM_STATE).copy()
    sample_df["cleaned_once"] = sample_df["text"].apply(lambda s: text_clean(s, opt_remove_stop, opt_mask_numbers))
    # apply again to prove idempotency (should be identical)
    sample_df["cleaned_twice"] = sample_df["cleaned_once"].apply(lambda s: text_clean(s, opt_remove_stop, opt_mask_numbers))
    st.dataframe(sample_df[["label", "text", "cleaned_once", "cleaned_twice"]], use_container_width=True)

st.caption("© Spam Classifier (LogReg) — deterministic preprocessing: lowercasing → masks(URL/EMAIL/PHONE/NUM) → punctuation removal → optional stopword removal (OFF by default).")

