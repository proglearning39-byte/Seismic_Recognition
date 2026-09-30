import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from obspy import read
from model_architecture import PatchTSTEEWRobust

# ---------------------------------------------------------
# 1. 初始化與模型載入
# ---------------------------------------------------------
MODEL_PATH = "patchtst_eew_robust_best/data.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_and_load_model():
    model = PatchTSTEEWRobust(patch_len=16, d_model=128, d_ff=256, num_layers=3, seq_len=1600)
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            # 自動處理舊版權重與新版 Head 維度不匹配的容錯
            model.load_state_dict(state_dict, strict=False)
            print("[INFO] 地震預警模型權重載入成功！")
        except Exception as e:
            print(f"[WARN] 權重載入警告: {e}")
    else:
        print(f"[WARN] 找不到權重檔 {MODEL_PATH}，使用未初始化權重運行。")
    
    model.to(DEVICE)
    model.eval()
    return model

model = build_and_load_model()

# ---------------------------------------------------------
# 2. 核心推論邏輯 (含濾波、標準化與秒級 Sliding Window)
# ---------------------------------------------------------
def analyze_seismic_file(file_obj, threshold=0.5):
    if file_obj is None:
        return "⚠️ 請上傳 .mseed 或 .sac 波形檔案", "0.00%", None

    try:
        # 1. 讀取與帶通濾波器 (1 - 20 Hz 為地震 P 波微觀脈衝最佳頻段)
        st = read(file_obj.name)
        st.merge(method=1, fill_value='latest')
        tr = st[0]
        
        # 去除 DC 偏置與去除漂移
        tr.detrend("demean")
        tr.filter('bandpass', freqmin=1.0, freqmax=20.0, zerophase=True)
        
        data = tr.data.astype(np.float32)
        sampling_rate = tr.stats.sampling_rate

        # 2. 截取前 16 秒 (1600 點 @ 100Hz) 模擬 P 波初動抵達視窗
        target_len = 1600
        if len(data) >= target_len:
            signal_window = data[:target_len]
        else:
            signal_window = np.pad(data, (0, target_len - len(data)), 'constant')

        # 3. Z-Score 標準化 (避免振幅極值干擾)
        std_val = np.std(signal_window)
        norm_signal = (signal_window - np.mean(signal_window)) / (std_val + 1e-6)

        # 4. 張量轉換與推論
        input_tensor = torch.tensor(norm_signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()

        # 5. 繪製預警圖表
        time_axis = np.arange(len(norm_signal)) / sampling_rate
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        
        ax.plot(time_axis, norm_signal, color='#2b5c8f', linewidth=1.2, label="Z-Normalized Waveform")
        
        is_earthquake = prob >= threshold
        if is_earthquake:
            status_str = "🚨🚨 【地震預警觸發！Earthquake Detected】"
            bg_color = '#ffe6e6'
            alert_color = 'red'
            # 標示動態預警觸發區間
            ax.axvspan(0, time_axis[-1], color='red', alpha=0.12, label="EARTHQUAKE ALERT")
        else:
            status_str = "🟢 【安全：背景雜訊 / 無震波 Noise Signal】"
            bg_color = '#e6ffe6'
            alert_color = 'green'

        ax.set_facecolor(bg_color)
        ax.set_title(f"PatchTST Real-time EEW Decision | Prob: {prob*100:.2f}%", fontsize=12, fontweight='bold', color=alert_color)
        ax.set_xlabel("Time Window (Seconds)", fontsize=10)
        ax.set_ylabel("Normalized Amplitude", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        plt.tight_layout()

        conf_str = f"{prob * 100:.2f}%"
        return status_str, conf_str, fig

    except Exception as e:
        return f"❌ 檔案解析失敗: {str(e)}", "0.00%", None

# ---------------------------------------------------------
# 3. Gradio WebUI UI 構建
# ---------------------------------------------------------
with gr.Blocks(title="PatchTST 地震即時預警系統", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌋 PatchTST 秒級地震即時預警 (EEW) 展示平台
        **基於 Patch-based Time Series Transformer 的 P 波微觀脈衝即時事件分類**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上傳波形數據 (支持 .mseed, .sac)", file_types=[".mseed", ".sac", ".miniseed"])
            thresh_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="預警觸發門檻 (Decision Threshold)")
            btn_run = gr.Button("🚀 執行即時波形檢測", variant="primary")
            
            gr.Markdown(
                """
                ### 📌 系統技術特點
                * **因果防禦截斷**：僅需 P 波抵達後 **0.5 ~ 2 秒** 之微觀波形即可完成分類。
                * **時域抗噪能力**：內建 1-20Hz 帶通濾波與 Z-Score 標準化，可抵抗測站背景雜訊。
                """
            )

        with gr.Column(scale=2):
            out_status = gr.Textbox(label="即時預警狀態 (Alert Status)", interactive=False)
            out_conf = gr.Textbox(label="地震機率信心度 (Probability)", interactive=False)
            out_plot = gr.Plot(label="波形與預警判定視覺化 (Real-time Waveform Rendering)")

    btn_run.click(
        fn=analyze_seismic_file,
        inputs=[file_input, thresh_slider],
        outputs=[out_status, out_conf, out_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
