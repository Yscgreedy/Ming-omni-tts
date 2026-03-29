import torch
print("torch version: ", torch.__version__)
print("cuda version: ", torch.version.cuda)
print("cudnn version: ", torch.backends.cudnn.version())
print("cuda available: ", torch.cuda.is_available())

try:
    import flash_attn
except:
    print("flash-attn not found")