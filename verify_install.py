# verify_install.py
import os
import sys, torch

print('Python  :', sys.version.split()[0])
print('PyTorch :', torch.__version__)

if torch.cuda.is_available():
    print('GPU     :', torch.cuda.get_device_name(0))
    try:
        torch.rand(1, device='cuda')
        print('VRAM    : OK')
    except Exception as e:
        print('VRAM    : FAILED ({})'.format(e))
else:
    print('GPU     : Not detected (CPU mode)')
    print('MKL     : {}'.format('OK' if torch.backends.mkl.is_available() else 'NOT FOUND'))

print('End')
os._exit(0)        