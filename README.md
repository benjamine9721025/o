# 📬 Spam Classifier — Streamlit Web Dashboard

這是一個使用 **Streamlit** 製作的互動式 **垃圾郵件分類器 (Spam Classifier)** 儀表板，  
可即時分析與預測簡訊是否為垃圾訊息，並提供可視化統計與模型效能分析。

---

## 🚀 專案簡介

本應用利用 **Logistic Regression (邏輯迴歸)** 模型進行文字分類，  
結合 **TF-IDF 特徵向量化** 與自訂文字正規化管線，  
在網頁上以互動方式呈現資料分析、模型評估與即時預測結果。
https://tbyoipjmgdxicfnaoqeaeh.streamlit.app/
---

## 🧩 功能特色

| 模組 | 功能說明 |
|------|-----------|
| 📂 **資料讀取** | 從 [GitHub datasets 目錄](https://github.com/benjamine9721025/o/tree/main/datasets) 自動讀取 `sms_spam_no_header.csv` |
| 🧹 **資料前處理 (Deterministic + Idempotent)** | <br>1️⃣ 全部轉小寫與空白正規化<br>2️⃣ 遮罩：網址→`<URL>`、Email→`<EMAIL>`、電話→`<PHONE>`、數字→`<NUM>`<br>3️⃣ 去除標點符號（保留特殊標記）<br>4️⃣ 可選停用詞移除（預設關閉） |
| 📊 **資料分析** | 顯示各類別比例與文字標記統計 |
| 🔠 **關鍵字可視化** | 顯示每類最常見 Top-N 詞彙 |
| 📈 **模型效能視覺化** | 混淆矩陣、ROC Curve、PR Curve、F1-score 表 |
| ⚡ **即時預測 (Live Inference)** | 使用者輸入文字後自動預測 spam/ham<br>顯示機率條圖與門檻線 τ，並提供一鍵範例載入 |
| 🧠 **模型** | Logistic Regression (TF-IDF 特徵) |
| 🛡️ **可重現性** | 固定 random_state 與確定性預處理，結果一致可再現 |

---

## 🏗️ 專案架構
📦 Spam-Classifier
├─ hw3.py ← 主應用程式 (Streamlit)
├─ requirements.txt ← 相依套件
└─ README.md ← 專案說明文件


## 🧠 模型與技術細節
演算法：Logistic Regression

特徵向量化：TF-IDF (1–2 grams)

訓練/測試分割：80 / 20

主要評估指標：F1-score, ROC-AUC, PR-AUC

預處理確定性：Deterministic & Idempotent Cleaning


## 📁 資料來源

Dataset: sms_spam_no_header.csv

無標題 CSV，第一欄為 label (ham / spam)，第二欄為 text


## 開發者

Author: @benjamine9721025

Project: Spam Classifier — AI & Information Security Course (NCHU)
Framework: Streamlit + Scikit-learn + Python 3.12