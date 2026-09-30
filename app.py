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
# 1. 訊號處理與模型載入函數
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
threshold = st.sidebar.slider("警報發布門檻 (Threshold)", 0.50, 0.99, 0.85, 0.01)

if "consecutive_triggers" not in st.session_state:
    st.session_state.consecutive_triggers = 0

# UI 主標題
st.title("🌋 Seismic PatchTST 秒級即時預警系統")
st.markdown("結合 **PatchTST Transformer** 與 **16s 滑動視窗 (100Hz)** 之雲端即時預警服務。")

# -----------------------------------------------------------------------------
# 3. 波形推論核心函數
# -----------------------------------------------------------------------------

def predict(raw_data, model_inst):
    # A. 帶通濾波 (去除低頻溫漂與高頻極端雜訊)
    filtered_data = bandpass_filter(raw_data)
    
    # B. 去平均與 Z-Score 標準化
    wave = filtered_data - np.mean(filtered_data)
    std = np.std(wave)
    normalized = wave / std if std > 1e-6 else np.zeros_like(wave)
    
    # C. 張量轉換與推論
    input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model_inst(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        prob_earthquake = probs[0, 1].item()  # 取得地震類別 (Index 1) 的信心度
        
    return normalized, prob_earthquake

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
        normalized_wave, prob = predict(valid_data, model)
        
        # 多訊框 Voting 防誤報邏輯
        if prob >= threshold:
            st.session_state.consecutive_triggers += 1
        else:
            st.session_state.consecutive_triggers = max(0, st.session_state.consecutive_triggers - 1)
            
        is_alarm = st.session_state.consecutive_triggers >= 3
        
        # 關鍵指標卡片
        col1, col2, col3 = st.columns(3)
        col1.metric("當前模型地震信心度", f"{prob*100:.2f}%")
        col2.metric("時間一致性計數 (Voting)", f"{st.session_state.consecutive_triggers}/3")
        col3.metric("連線狀態", "🟢 Live" if is_live else "🟡 Fallback (Simulated)")
        
        # 警報橫幅
        if is_alarm:
            st.error(f"🚨 [EARTHQUAKE WARNING] 偵測到地震 P 波衝擊！信心度: {prob*100:.2f}%")
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
        
        # 若上傳資料點數不足或超過，進行預設擷取
        if len(raw_data) >= 1600:
            raw_data = raw_data[:1600]
            normalized_wave, prob = predict(raw_data, model)
            
            if prob >= threshold:
                st.error(f"🚨 警報：判定為地震波！ (信心度: {prob*100:.2f}%)")
            else:
                st.success(f"🟢 正常背景波形 (信心度: {prob*100:.2f}%)")
                
            time_axis = np.linspace(0, 16, 1600)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(time_axis, normalized_wave, color='crimson' if prob >= threshold else '#1f77b4', lw=1.0)
            ax.set_title("Playback Waveform (16s @ 100Hz)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized Amp")
            ax.set_xlim(0, 16)
            ax.set_ylim(-5, 5)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("❌ 上傳檔案之波形點數不足 1600 點 (16 秒)！")
