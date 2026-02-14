import os
import torch
import torchaudio
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)

from models.ecapa import ECAPA_TDNN
from utils.meters import _l2norm

# ========================= 配置 =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "../outputs/best.pt"
SAMPLE_RATE = 16000
CROP_FRAMES = 400
NUM_CROPS = 6
SIM_THRESHOLD = 0.65

# ========================= 模型加载 =========================
print("Loading model...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = ECAPA_TDNN(
    in_channels=80,
    channels=ckpt.get("channels", 512),
    embd_dim=256
).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# ========================= 音频处理函数（不变） =========================
def load_and_process_audio(audio_path: str):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
        n_mels=80, f_min=20, f_max=8000, norm="slaney", mel_scale="slaney"
    )(waveform.squeeze(0))
    fbank = torch.log(fbank + 1e-6).transpose(0, 1)
    return fbank

@torch.no_grad()
def extract_embedding(audio_path: str):
    feat = load_and_process_audio(audio_path)
    T = feat.size(0)
    if T <= CROP_FRAMES:
        x = feat.unsqueeze(0).to(DEVICE)
        emb = model(x).squeeze(0).cpu()
    else:
        embs = []
        for _ in range(NUM_CROPS):
            start = np.random.randint(0, T - CROP_FRAMES)
            chunk = feat[start:start + CROP_FRAMES]
            x = chunk.unsqueeze(0).to(DEVICE)
            embs.append(model(x).squeeze(0).cpu())
        emb = torch.stack(embs, 0).mean(0)
    return _l2norm(emb)

# ========================= 验证函数 =========================
def verify_speakers(audio1, audio2):
    if audio1 is None or audio2 is None:
        return "请上传两个音频！", 0.0, None, None
    try:
        emb1 = extract_embedding(audio1)
        emb2 = extract_embedding(audio2)
        sim = float((emb1 * emb2).sum().item())
        is_same = sim > SIM_THRESHOLD
        result_text = "✅ **同一说话人**" if is_same else "❌ **不同说话人**"
        
        def plot_waveform(audio_path, title):
            waveform, _ = torchaudio.load(audio_path)
            waveform = waveform.mean(0).numpy()
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(waveform)
            ax.set_title(title)
            ax.grid(True)
            return fig
        
        return (
            f"{result_text}\n**相似度**: {sim:.4f}",
            sim,
            plot_waveform(audio1, "Speaker 1"),
            plot_waveform(audio2, "Speaker 2")
        )
    except Exception as e:
        return f"错误: {str(e)}", 0.0, None, None

with gr.Blocks(title="说话人验证 Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Speaker Verification 在线 Demo")
    gr.Markdown("**基于 ECAPA-TDNN + AAM-Softmax** | 支持麦克风实时录音")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 说话人 1")
            audio1 = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="上传或录音"
            )
        
        with gr.Column():
            gr.Markdown("### 说话人 2")
            audio2 = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="上传或录音"
            )
    
    with gr.Row():
        btn = gr.Button("🔍 开始验证", variant="primary", size="large")
    
    with gr.Row():
        with gr.Column(scale=2):
            result = gr.Markdown(label="验证结果", value="等待验证...")
        with gr.Column():
            score = gr.Number(label="相似度分数", value=0.0)
    
    with gr.Row():
        waveform1 = gr.Plot(label="说话人 1 波形")
        waveform2 = gr.Plot(label="说话人 2 波形")
    
    gr.Markdown("---\n**使用说明**：\n"
                "1. 上传两个 .wav 文件（或直接录音）\n"
                "2. 点击「开始验证」\n"
                "3. 相似度 > 0.65 判定为同一说话人")

    # 绑定
    btn.click(
        fn=verify_speakers,
        inputs=[audio1, audio2],
        outputs=[result, score, waveform1, waveform2]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=True
    )