import torch

# 新增设备验证代码
print(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")
print(f"可用 CUDA 设备: {torch.cuda.device_count()}")
print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 测试 GPU 计算
if torch.cuda.is_available():
    x = torch.tensor([1.0]).cuda()
    print(f"\nGPU 张量计算结果: {x * 2}")
else:
    print("\n⚠️ 未检测到可用 GPU，请检查：")
    print("1. 是否已安装 NVIDIA 显卡驱动")
    print("2. CUDA 工具包版本是否与 PyTorch 匹配")
    print("3. 是否安装正确版本的 PyTorch GPU 版")