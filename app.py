import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from streamlit_autorefresh import st_autorefresh
from model import PatchTSTEEWRobust
from seedlink_client import fetch_latest_waveform

# 頁面標題配置
st.set_page_config(page_title="PatchTST EEW 地震即時預警系統", layout="wide", page_icon="🌋")

# 1. 載入模型 (利用 @st.cache_resource 避免重複載入)
@st.cache_resource
def load_model():
    model = PatchTSTEEWRobust()
    model_path = "patchtst_eew_best.pt"

    if not os.path.exists(model_path):
        st.error(f"❌ 找不到模型權重檔案: {model_path}，請確認是否已上傳至 GitHub！")
        st.stop()
        
    try:
        model.load_state_dict(torch.load("patchtst_eew_best.pt", map_location="cpu"))
        print("✅ 成功載入模型權重檔！")
    except Exception as e:
        st.error(f"❌ 模型權重載入失敗: {e}")
        st.stop()
        
    model.eval()
    return model

model = load_model()

# 2. 視窗側邊欄控制
st.sidebar.title("⚙️ 預警系統控制台")
mode = st.sidebar.radio("模式選擇", ["🛰️ 即時串流監測 (Live SeedLink)", "📁 歷史波形重播測試 (Playback)"])
threshold = st.sidebar.slider("警報發布門檻 (Threshold)", 0.50, 0.99, 0.85, 0.01)

# 初始化 Session State 防誤報投票數
if "consecutive_triggers" not in st.session_state:
    st.session_state.consecutive_triggers = 0

# UI 主畫面
st.title("🌋 Seismic PatchTST 秒級即時預警系統")
st.markdown("結合 **PatchTST Transformer** 與 **100s 滑動視窗** 之雲端即時預警服務。")

# 波形預處理與推論函式
def predict(raw_data):
    # 建議：若訓練時沒有做 Bandpass 濾波，推論時也不該做！
    # 僅做基礎 去平均 (Demean) 與 Z-Score 正規化
    wave = raw_data - np.mean(raw_data)
    std = np.std(wave)
    normalized = (wave) / std if std > 1e-6 else np.zeros_like(wave)
    
    # 轉為 PyTorch Tensor: shape (batch_size=1, channel=1, seq_len=1600)
    input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
        # 取得機率值
        prob = torch.sigmoid(logits).item()
    return normalized, prob

if mode == "🛰️ 即時串流監測 (Live SeedLink)":
    # 每一秒刷新一次 (1000ms)
    count = st_autorefresh(interval=2000, key="eew_refresh")
    
    st.subheader("🛰️ IRIS ANMO 測站實時連線中...")
    data, is_live = fetch_latest_waveform()
    normalized_wave, prob = predict(data)
    
    # Voting 邏輯
    if prob >= threshold:
        st.session_state.consecutive_triggers += 1
    else:
        st.session_state.consecutive_triggers = max(0, st.session_state.consecutive_triggers - 1)
        
    is_alarm = st.session_state.consecutive_triggers >= 3
    
    # 指標顯示
    col1, col2, col3 = st.columns(3)
    col1.metric("當前模型地震信心度", f"{prob*100:.2f}%")
    col2.metric("時間一致性計數 (Voting)", f"{st.session_state.consecutive_triggers}/3")
    col3.metric("連線狀態", "🟢 Live" if is_live else "🟡 Fallback (Simulated)")
    
    if is_alarm:
        st.error(f"🚨 [EARTHQUAKE WARNING] 偵測到地震 P 波衝擊！信心度: {prob*100:.2f}%")
    else:
        st.success("🟢 系統監測中：未偵測到地震特徵")
        
    # 繪製圖表
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(normalized_wave, color='crimson' if is_alarm else 'navy', lw=1.2)
    ax.set_title("Real-time Waveform (16s @ 100Hz)")
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

else:
    st.subheader("📁 歷史地震波形模擬測試")
    uploaded_file = st.file_uploader("上傳 .npy 波形檔 (1600 點 @ 100Hz)", type=["npy"])
    
    if uploaded_file is not None:
        raw_data = np.load(uploaded_file)
        normalized_wave, prob = predict(raw_data)
        
        if prob >= threshold:
            st.error(f"🚨 警報：判定為地震波！ (信心度: {prob*100:.2f}%)")
        else:
            st.success(f"🟢 正常背景波形 (信心度: {prob*100:.2f}%)")
            
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(normalized_wave, color='crimson' if prob >= threshold else 'navy', lw=1.2)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
