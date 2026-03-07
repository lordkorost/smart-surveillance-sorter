lines = open('requirements.txt').readlines()
filtered = [l for l in lines if not l.lower().startswith(
    ('torch','torchvision','torchaudio','pytorch-triton','rocm'))]
open('_req_tmp.txt', 'w').writelines(filtered)