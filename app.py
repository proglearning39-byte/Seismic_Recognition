import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from obspy import read
from model_architecture import PatchTSTEEWRobust

# ---------------------------------------------------------
# 1. 模型載入與設定
# ---------------------------------------------------------
MODEL_PATH = "patchtst_eew_robust_best/data.pkl"  # 您的權重路徑
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_eew_model():
    model = PatchTSTEEWRobust()
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("Successfully loaded model weights!")
        except Exception as e:
            print(f"Error loading state_dict: {e}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Running with uninitialized model.")
    model.to(DEVICE)
    model.eval()
    return model

model = load_eew_model()

# ---------------------------------------------------------
# 2. 波形推論核心邏輯
# ---------------------------------------------------------
def process_seismic_signal(file_obj, patch_length_sec=0.5, sampling_rate=100.0):
    """
    處理上傳的波形檔案，進行波形預處理、PatchTST 模型推論與數據繪製
    """
    if file_obj is None:
        return "請上傳 MSEED / SAC 地震檔案", None, None

    try:
        # 讀取地震波形
        st = read(file_obj.name)
        st.filter('bandpass', freqmin=1.0, freqmax=20.0) # 標準 1-20Hz 帶通濾波器
        tr = st[0]
        data = tr.data.astype(np.float32)

        # 標準化 (Z-score Normalization)
        data = (data - np.mean(data)) / (np.std(data) + 1e-6)

        # 自動裁切/補齊至模型預期的輸入長度（例如 1600 個採樣點 = 16 秒）
        target_length = 1600 
        if len(data) > target_length:
            # 擷取波形前半段或前 16 秒
            input_data = data[:target_length]
        else:
            # Padding 補零
            input_data = np.pad(data, (0, target_length - len(data)), 'constant')

        tensor_input = torch.tensor(input_data).unsqueeze(0).to(DEVICE)

        # 模型推論 (Inference)
        with torch.no_grad():
            logits = model(tensor_input)
            prob = torch.sigmoid(logits).item()

        # 警報判定 (Threshold = 0.5)
        is_earthquake = prob >= 0.5
        status_color = "🔴 【強烈地震預警！】" if is_earthquake else "🟢 【安全：背景雜訊 / 無震波】"
        confidence_text = f"地震事件信心度 (Earthquake Probability): {prob * 100:.2f}%"

        # 繪製波形與二元預警檢測圖表
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)
        time_axis = np.arange(len(input_data)) / sampling_rate
        ax.plot(time_axis, input_data, color='#1f77b4', linewidth=1.0, label="Filtered Seismic Velocity")
        
        if is_earthquake:
            ax.set_facecolor('#fff0f0') # 發生地震時顯示紅底警示區域
            ax.axvspan(0, time_axis[-1], color='red', alpha=0.15, label="Earthquake Triggered")
        else:
            ax.set_facecolor('#f4fbf4')

        ax.set_title(f"Seismic Waveform Analysis | Prediction: {'EARTHQUAKE' if is_earthquake else 'NOISE'}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Normalized Amplitude", fontsize=10)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        return status_color, confidence_text, fig

    except Exception as e:
        return f"波形解析失敗: {str(e)}", "", None

# ---------------------------------------------------------
# 3. Gradio 介面搭建 (UI Interface)
# ---------------------------------------------------------
custom_css = """
#alert-box { font-size: 20px; font-weight: bold; text-align: center; }
#conf-box { font-size: 16px; text-align: center; }
"""

with gr.Blocks(css=custom_css, title="Seismic PatchTST Real-time Early Warning") as demo:
    gr.Markdown(
        """
        # 🌋 地震深度學習即時預警系統 (PatchTST EEW System)
        **AI 驅動秒級地震波分類與 P 波微觀脈衝識別**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上傳地震資料檔 (.mseed, .sac, .miniseed)", file_types=[".mseed", ".sac", ".miniseed"])
            submit_btn = gr.Button("🚨 執行波形檢測與推論", variant="primary")
            
            gr.Markdown(
                """
                ---
                ### 💡 專案說明與特點
                1. **PatchTST 時序架構**：將時域連續波形進行 Patch 化，精準掌握 P 波初動特徵。
                2. **因果防禦機制**：針對 P 波抵達後 0.5~3 秒極短視窗進行微觀動態訓練。
                """
            )
            
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="系統即時狀態 (Status)", elem_id="alert-box")
            confidence_output = gr.Textbox(label="AI 預測信心水準 (Confidence)", elem_id="conf-box")
            plot_output = gr.Plot(label="測站動態時域波形圖 (Waveform Rendering)")

    submit_btn.click(
        fn=process_seismic_signal,
        inputs=[file_input],
        outputs=[status_output, confidence_output, plot_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
