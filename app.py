import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import butter, filtfilt
from streamlit_autorefresh import st_autorefresh

from model import PatchTSTEEWRobust
from seedlink_client import fetch_latest_waveform

# 頁面標題與佈局配置
st.set_page_config(page_title="PatchTST EEW 地震即時預警系統", layout="wide", page_icon="🌋")

# -----------------------------------------------------------------------------
# 1. 訊號處理與防誤報演算法 (STA/LTA + 帶通濾波)
# -----------------------------------------------------------------------------

def bandpass_filter(data, lowcut=1.0, highcut=20.0, fs=100.0, order=4):
    """1~20Hz 帶通濾波器：濾除 DC bias 與基底低頻溫漂"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if len(data) > 3 * order:
        return filtfilt(b, a, data)
    return data

def calc_sta_lta(data, sta_len=50, lta_len=500):
    """
    計算地震學經典 STA/LTA (短短/長平均能量比)
    - sta_len=50 (0.5秒)
    - lta_len=500 (5.0秒)
    """
    data_sq = data ** 2
    # 近期短時間能量
    sta = np.mean(data_sq[-sta_len:])
    # 長時間背景能量
    lta = np.mean(data_sq[-lta_len:])
    
    if lta < 1e-8:
        return 0.0
    return sta / lta

@st.cache_resource
def load_model():
    """載入 PatchTST 模型與權重"""
    model = PatchTSTEEWRobust()
    model_path = "patchtst_eew_best.pt"

    if not os.path.exists(model_path):
        st.error(f"❌ 找不到模型權重檔案: {model_path}，請確認是否已上傳至 GitHub！")
        st.stop()
        
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            
        model.load_state_dict(checkpoint, strict=True)
        print("🎉 恭喜！模型權重 100% 精準匹配成功！")
    except Exception as e:
        st.error(f"❌ 模型權重載入失敗: {e}")
        st.stop()
        
    model.eval()
    return model

model = load_model()

# -----------------------------------------------------------------------------
# 2. UI 側邊欄與 session_state 初始化
# -----------------------------------------------------------------------------

st.sidebar.title("⚙️ 預警系統控制台")
mode = st.sidebar.radio("模式選擇", ["🛰️ 即時串流監測 (Live SeedLink)", "📁 歷史波形重播測試 (Playback)"])
threshold = st.sidebar.slider("AI 警報門檻 (Threshold)", 0.50, 0.99, 0.85, 0.01)
sta_lta_threshold = st.sidebar.slider("STA/LTA 能量突變門檻", 1.5, 8.0, 3.0, 0.1)

if "consecutive_triggers" not in st.session_state:
    st.session_state.consecutive_triggers = 0

# UI 主標題
st.title("🌋 Seismic PatchTST 秒級即時預警系統")
st.markdown("結合 **PatchTST Transformer** 與 **STA/LTA 混合觸發 (Hybrid Trigger)** 之即時預警服務。")

# -----------------------------------------------------------------------------
# 3. 波形推論核心函數 (含雙重物理防線)
# -----------------------------------------------------------------------------

def predict(raw_data, model_inst):
    # A. 帶通濾波 (去除低頻溫漂與高頻極端雜訊)
    filtered_data = bandpass_filter(raw_data)
    
    # B. 計算物理特徵：絕對標準差與 STA/LTA
    raw_std = np.std(filtered_data)
    sta_lta_ratio = calc_sta_lta(filtered_data)
    
    # C. 去平均與 Z-Score 標準化
    wave = filtered_data - np.mean(filtered_data)
    std = np.std(wave)
    normalized = wave / std if std > 1e-6 else np.zeros_like(wave)
    
    # -------------------------------------------------------------------------
    # 🛡️ 防線 1：絕對能量硬過濾 (當前波形全為背景平靜雜訊時，直接阻斷 AI 推論)
    # -------------------------------------------------------------------------
    if raw_std < 1.0:  # 數值可依 SeedLink 測站單位調整
        return normalized, 0.0001, sta_lta_ratio
        
    # D. 送入 PatchTST 模型進行深度推論
    input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model_inst(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        prob_earthquake = probs[0, 1].item()
        
    return normalized, prob_earthquake, sta_lta_ratio

# -----------------------------------------------------------------------------
# 4. 主監測與測試邏輯
# -----------------------------------------------------------------------------

if mode == "🛰️ 即時串流監測 (Live SeedLink)":
    # 設置 2 秒自動刷新
    st_autorefresh(interval=2000, key="eew_refresh")
    
    st.subheader("🛰️ IRIS ANMO 測站實時連線中...")
    data, is_live = fetch_latest_waveform()
    
    # 防護機制：確認 SeedLink Buffer 是否已積滿 16 秒 (1600 點)
    if data is None or len(data) < 1600:
        st.warning(f"⏳ 即時波形緩衝區累積中... ({len(data) if data is not None else 0} / 1600 點)，請稍候數秒以確保波形完整。")
    else:
        # 取最新 1600 個真實資料點，防止補零 (Zero-padding) 造成斷崖波形
        valid_data = np.array(data[-1600:])
        normalized_wave, prob, sta_lta_ratio = predict(valid_data, model)
        
        # ---------------------------------------------------------------------
        # 🛡️ 防線 2：AI 信心度 + STA/LTA 能量突變雙重驗證 (Hybrid Triggering)
        # ---------------------------------------------------------------------
        is_hybrid_triggered = (prob >= threshold) and (sta_lta_ratio >= sta_lta_threshold)
        
        if is_hybrid_triggered:
            st.session_state.consecutive_triggers += 1
        else:
            st.session_state.consecutive_triggers = max(0, st.session_state.consecutive_triggers - 1)
            
        is_alarm = st.session_state.consecutive_triggers >= 3
        
        # 關鍵指標卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("模型地震信心度", f"{prob*100:.2f}%")
        col2.metric("STA/LTA 突變比", f"{sta_lta_ratio:.2f}", delta="觸發門檻 ≥ " + str(sta_lta_threshold))
        col3.metric("時間一致性 (Voting)", f"{st.session_state.consecutive_triggers}/3")
        col4.metric("連線狀態", "🟢 Live" if is_live else "🟡 Fallback (Simulated)")
        
        # 警報橫幅
        if is_alarm:
            st.error(f"🚨 [EARTHQUAKE WARNING] 雙驗證偵測到地震 P 波衝擊！信心度: {prob*100:.2f}%, STA/LTA: {sta_lta_ratio:.2f}")
        elif prob >= threshold and sta_lta_ratio < sta_lta_threshold:
            st.warning(f"⚠️ [雜訊過濾] AI 判斷概率高 ({prob*100:.2f}%)，但無物理能量突變 (STA/LTA: {sta_lta_ratio:.2f}) -> 認定為放大雜訊！")
        else:
            st.success("🟢 系統監測中：未偵測到地震特徵")
            
        # 繪製 Matplotlib 即時波形 (將 X 軸轉為 0~16 秒)
        time_axis = np.linspace(0, 16, 1600)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(time_axis, normalized_wave, color='crimson' if is_alarm else '#1f77b4', lw=1.0)
        ax.set_title("Real-time Waveform (16s @ 100Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Amp")
        ax.set_xlim(0, 16)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

else:
    st.subheader("📁 歷史地震波形模擬測試")
    uploaded_file = st.file_uploader("上傳 .npy 波形檔 (1600 點 @ 100Hz)", type=["npy"])
    
    if uploaded_file is not None:
        raw_data = np.load(uploaded_file)
        
        if len(raw_data) >= 1600:
            raw_data = raw_data[:1600]
            normalized_wave, prob, sta_lta_ratio = predict(raw_data, model)
            
            is_triggered = (prob >= threshold) and (sta_lta_ratio >= sta_lta_threshold)
            
            if is_triggered:
                st.error(f"🚨 警報：判定為地震波！ (信心度: {prob*100:.2f}%, STA/LTA: {sta_lta_ratio:.2f})")
            else:
                st.success(f"🟢 正常背景波形 (信心度: {prob*100:.2f}%, STA/LTA: {sta_lta_ratio:.2f})")
                
            time_axis = np.linspace(0, 16, 1600)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(time_axis, normalized_wave, color='crimson' if is_triggered else '#1f77b4', lw=1.0)
            ax.set_title("Playback Waveform (16s @ 100Hz)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized Amp")
            ax.set_xlim(0, 16)
            ax.set_ylim(-5, 5)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("❌ 上傳檔案之波形點數不足 1600 點 (16 秒)！")
