import sys, torch
print(f'Python  : {sys.version.split()[0]}')
print(f'PyTorch : {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    try:
        torch.rand(1, device='cuda')
        print('VRAM    : OK')
    except Exception as e:
        print(f'VRAM    : FAILED ({e})')
else:
    print('GPU     : Not detected (CPU mode)')
    print(f'MKL     : {"OK" if torch.backends.mkl.is_available() else "NOT FOUND"}')