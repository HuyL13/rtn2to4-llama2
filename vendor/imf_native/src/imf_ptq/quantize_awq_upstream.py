import argparse, gc, json, sys
from pathlib import Path
from .calibration import load_calibration, token_blocks
from .provenance import git_identity, sha256_file

AWQ_URL="https://github.com/mit-han-lab/llm-awq"
def awq_metadata(bits:int, group_size:int, commit:str, calibration_sha256:str)->dict:
    return {"quantizer":"AWQ","bits":bits,"group_size":group_size,"quantizer_commit":commit,
            "calibration_sha256":calibration_sha256,"packed_int3_runtime":False,
            "storage_representation":"HF dequantized quantized-on-grid weights"}

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--upstream",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--calibration",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--bits",type=int,choices=[3,4],required=True); p.add_argument("--group-size",type=int,default=128)
    a=p.parse_args(); identity=git_identity(a.upstream,AWQ_URL); sys.path.insert(0,str(a.upstream))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from awq.quantize.pre_quant import run_awq, apply_awq
    from awq.quantize.quantizer import pseudo_quantize_model_weight
    import awq.utils.calib_data as calibration_module
    model=AutoModelForCausalLM.from_pretrained(a.source,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation="flash_attention_2").eval(); model.config.use_cache=False; tok=AutoTokenizer.from_pretrained(a.source)
    samples=load_calibration(a.calibration)
    blocks=token_blocks(samples,tok,512,len(samples)); calibration_module.get_calib_dataset=lambda **_: blocks
    q_config={"zero_point":True,"q_group_size":a.group_size}; awq=run_awq(model,tok,w_bit=a.bits,q_config=q_config,n_samples=len(blocks),seqlen=512,calib_data="fixed")
    # Match upstream's --run_awq/--load_awq two-process flow: apply results to a pristine reload.
    del model; gc.collect(); torch.cuda.empty_cache()
    model=AutoModelForCausalLM.from_pretrained(a.source,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
    apply_awq(model,awq); pseudo_quantize_model_weight(model,w_bit=a.bits,q_config=q_config)
    a.output.mkdir(parents=True,exist_ok=True); model.save_pretrained(a.output,safe_serialization=True); tok.save_pretrained(a.output)
    meta=awq_metadata(a.bits,a.group_size,identity["commit"],sha256_file(a.calibration)); meta["checkpoint_path"]=str(a.output); (a.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__": main()
