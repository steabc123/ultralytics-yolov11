import torch
import torchvision


# 指定 device 为 CPU 或 GPU
# 指定 CPU
# cpu1 =torch.device("cpu:0")
# print("CPU Device:【{}:{}】".format(cpu1.type,cpu1.index))

# 指定 GPU
gpu =torch.device("cuda:0")
print("GPU Device:【{}:{}】".format(gpu.type,gpu.index))

# 查看GPU是否可用及设备名称
print("Total GPU Count:{}".format(torch.cuda.device_count()))   #查看所有可用GPU个数
print("Total CPU Count:{}".format(torch.cuda.os.cpu_count()))   #获取系统CPU数量
print(torch.cuda.get_device_name(torch.device("cuda:0")))       #获取GPU设备名称   NVIDIA GeForce GT 1030
print("GPU Is Available:{}".format(torch.cuda.is_available()))  #GPU设备是否可用  True

print(torch.cuda.is_available())
print(f"Torch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")

# 在GPU上建立Tensor

# 建立 Tensor
device = torch.device('cuda:0')

# 使用 tensor 在 cpu 建立再转移到 gpu
gpu_tensor1= torch.Tensor([[1,4,7],[3,6,9],[2,5,8]]).to(device)  # 使用to()方法将cup_tensor转到GPU上
# 直接在 gpu 上建立
gpu_tensor2 = torch.tensor([[1,4,7],[3,6,9],[2,5,8]],device=device)
gpu_tensor3 = torch.rand((3,4),device=device)                           # 方法一 直接申明 device
gpu_tensor4  = torch.randn(3,4).float().to(device)                      # 方法二 使用 to device 转移

print(gpu_tensor1)
print(gpu_tensor2)
print(gpu_tensor3)
print(gpu_tensor4)

# 查看内存大小和显存信息
torch.cuda.empty_cache()                      # 释放没有使用的缓存数据

# print(torch.cuda.memory_cached())             # 获取缓存数据大小
# FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
#   warnings.warn(
print(torch.cuda.memory_reserved())
# FutureWarning: torch.cuda.max_memory_cached has been renamed to torch.cuda.max_memory_reserved
#   warnings.warn(
# print(torch.cuda.max_memory_cached())         # 最大缓存大小
print(torch.cuda.max_memory_reserved())
print(torch.cuda.max_memory_allocated())       # 最大分配内存大小
print(torch.cuda.memory_summary())            # 查看显存信息

# 如果GPU设备可用，将默认热备改为GPU
# 创建默认的CPU设备
# device = torch.device("cpu")
# 如果GPU设备可用，将默认设备改为GPU
if(torch.cuda.is_available()):
    device = torch.device("cuda")
