"""Explicit Colab setup for TFA IF-SFT, protecting installed Torch/CUDA."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from importlib.metadata import distributions

ROOT = Path(__file__).resolve().parents[1]


def protected(name):
    name = re.sub(r'[-_.]+', '-', name.lower())
    return name in {'torch','torchvision','torchaudio','triton'} or (
        name.startswith('nvidia-') and name != 'nvidia-ml-py')


def versions():
    return {re.sub(r'[-_.]+','-',d.metadata['Name'].lower()):d.version
            for d in distributions() if protected(d.metadata['Name'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('This setup is for the Linux Colab runtime only.')
    os.environ.update(USE_TF='0',USE_FLAX='0',USE_TORCH='1',MAX_JOBS='2',DS_BUILD_OPS='0')
    def run(*command):
        subprocess.run([str(x) for x in command],check=True,cwd=ROOT)
    original = versions()
    if 'torch' not in original:
        raise RuntimeError('Use a Colab GPU runtime with Torch already installed.')
    if not args.check_only:
        run('apt-get','-qq','update')
        run('apt-get','-qq','install','-y','--no-install-recommends','build-essential','libaio-dev')
        with tempfile.TemporaryDirectory(prefix='tfa_setup_') as directory:
            directory=Path(directory)
            constraints=directory/'keep_torch.txt'
            constraints.write_text('\n'.join(f'{n}=={v}' for n,v in original.items()))
            def install(arguments, name):
                command=[sys.executable,'-m','pip','install','--no-cache-dir','-c',constraints,*arguments]
                report=directory/name
                run(*command,'--dry-run','--report',report)
                forbidden=[x['metadata']['name'] for x in json.loads(report.read_text())['install']
                           if protected(x['metadata']['name'])]
                if forbidden:
                    raise RuntimeError(f'Refusing to change Torch/CUDA: {forbidden}')
                run(*command)
                if versions()!=original:
                    raise RuntimeError('Protected Torch/CUDA package versions changed.')
            install(['--only-binary=:all:','-r',ROOT/'requirements-tfa-if-sft.txt'],'dependencies.json')
            install(['--no-deps','--no-build-isolation','deepspeed==0.18.8'],'deepspeed.json')
    # A fresh interpreter avoids importing packages cached before pip ran.
    run(sys.executable,'-c',r'''
import json, tempfile
from pathlib import Path
import torch, transformers, accelerate, sentencepiece, tensorboard, deepspeed
from deepspeed.ops.op_builder import CPUAdamBuilder, AsyncIOBuilder
from scripts.train_tfa_if_sft import load_upstream, verify_upstream, DATA, BASE
assert transformers.__version__ == '4.45.0'
assert accelerate.__version__ == '1.1.1'
assert deepspeed.__version__ == '0.18.8'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
verify_upstream()
module=load_upstream('colab_nvme')
tokenizer=transformers.AutoTokenizer.from_pretrained(BASE,use_fast=False,padding_side='right',model_max_length=512)
assert tokenizer.pad_token_id is not None, 'Mirror tokenizer changed; inspect ZeRO-3 embedding resize before training.'
dataset=module.SupervisedDataset(str(DATA),tokenizer)
rows=json.loads(DATA.read_text(encoding='utf-8'))
fp=[i for i,row in enumerate(rows) if 'fingerprint' in row['output']]
assert len(dataset)==60 and len(fp)==10
assert all((dataset[i]['labels']!=-100).any() for i in fp)
CPUAdamBuilder().load()
aio=AsyncIOBuilder().load().aio_handle()
buffers=[torch.zeros(131084288,dtype=torch.float32).pin_memory() for _ in range(4)]
with tempfile.TemporaryDirectory(dir='/content') as directory:
    path=str(Path(directory)/'aio.bin')
    buffers[0].fill_(.125)
    assert aio.async_pwrite(buffers[0],path,0)==0 and aio.wait()==1
    assert aio.async_pread(buffers[1],path,0)==0 and aio.wait()==1
    assert torch.equal(buffers[0],buffers[1])
print('PASS upstream hashes/imports, 60 rows/10 supervised fingerprints, CPUAdam build, 4x500MiB buffers and AIO; Torch:',torch.__version__)
''')
    if versions()!=original:
        raise RuntimeError('Torch/CUDA changed during setup.')
    print('Setup complete; Torch/CUDA versions preserved.')


if __name__=='__main__':
    main()
