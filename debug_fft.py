import torch
import torch.fft
import time

def test_gpu_fft():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please check your driver.")
        return

    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # 模拟 VFPT 中的数据尺寸 (Batch=64, Tokens=10, Dim=768)
    # 这里的尺寸要和代码里报错时的一致
    x = torch.randn(64, 10, 768).cuda()
    
    print("\n🚀 Testing FFT on GPU...")
    try:
        start = time.time()
        # 这就是 vit_fourier.py 里的那行核心代码
        # x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        
        # 拆解动作，模拟 FNetBlock
        x_fft = torch.fft.fft(x, dim=-1)
        x_out = torch.fft.fft(x_fft, dim=-2).real
        
        torch.cuda.synchronize() # 等待 GPU 计算完成
        print(f"✅ Success! FFT Time: {(time.time() - start)*1000:.2f} ms")
        print("🎉 你的环境已完美支持 RTX 4090 跑 FFT！")
        
    except RuntimeError as e:
        print("\n❌ Failed! 依然报错:")
        print(e)
        print("\n结论: 这个环境还是不行，需要换 CUDA 版本。")

if __name__ == "__main__":
    test_gpu_fft()