import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
from .calibration import load_calibration, token_blocks
from .provenance import git_identity, sha256_file

GPTQ_URL="https://github.com/IST-DASLab/gptq"
def validate_gptq(bits:int,group_size:int)->tuple[int,int]:
    if (bits,group_size)!=(3,128): raise ValueError("only GPTQ3-G128 is permitted")
    return bits,group_size
def gptq_metadata(commit:str,calibration_sha256:str,options:dict|None=None)->dict:
    values=options or {"act_order":False,"true_sequential":True,"static_groups":False,"percdamp":0.01,"sym":False}
    return {"quantizer":"GPTQ","bits":3,"group_size":128,"quantizer_commit":commit,"calibration_sha256":calibration_sha256,"options":values,"storage_representation":"HF dequantized quantized-on-grid weights"}
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--upstream",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--calibration",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--bits",type=int,default=3); p.add_argument("--group-size",type=int,default=128); a=p.parse_args(); validate_gptq(a.bits,a.group_size)
    identity=git_identity(a.upstream,GPTQ_URL); manifest=json.loads(a.manifest.read_text()); options={k:manifest[k] for k in ["act_order","true_sequential","static_groups","percdamp","sym"]}
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    sys.path.insert(0,str(a.upstream)); import llama as upstream_llama
    tok=AutoTokenizer.from_pretrained(a.source,use_fast=True); model=AutoModelForCausalLM.from_pretrained(a.source,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True); model.seqlen=int(manifest["sequence_length"])
    samples=load_calibration(a.calibration); tensors=token_blocks(samples,tok,model.seqlen,len(samples)); blocks=[(x,x.clone()) for x in tensors]
    upstream_llama.args=SimpleNamespace(nsamples=len(blocks),wbits=3,groupsize=128,**options)
    device=torch.device("cuda:0")
    # Transformers >=4.45 made RoPE a model-level module; the older upstream
    # loader moves embeddings/norm itself but not this newly introduced module.
    model.model.rotary_emb=model.model.rotary_emb.to(device)
    upstream_llama.llama_sequential(model,blocks,device)
    model.model.rotary_emb=model.model.rotary_emb.cpu()
    a.output.mkdir(parents=True,exist_ok=True); model.save_pretrained(a.output,safe_serialization=True); tok.save_pretrained(a.output)
    meta=gptq_metadata(identity["commit"],sha256_file(a.calibration),options); meta["checkpoint_path"]=str(a.output); (a.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__": main()
