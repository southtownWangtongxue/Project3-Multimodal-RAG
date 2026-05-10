"""
使用显卡跑embedding模型，卸载cpu版本torch
pip uninstall torch torchvision torchaudio -y
安装gpu版torch，cu130为版本号
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
"""
import torch;


print(f'版本: {torch.__version__}');
print(f'CUDA可用: {torch.cuda.is_available()}');
print(f'设备数: {torch.cuda.device_count()}')
