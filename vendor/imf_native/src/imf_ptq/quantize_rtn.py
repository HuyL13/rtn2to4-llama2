import argparse
import json
import sys
from pathlib import Path
from .provenance import git_identity, sha256_file

GPTQ_URL = "https://github.com/IST-DASLab/gptq"

def resolve_if_rtn_backend(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError("completed IF-SFT RTN backend was not found")
    return path.resolve()

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--backend", type=Path); parser.add_argument("--upstream", type=Path, default=Path("third_party/gptq")); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--bits", type=int, required=True); parser.add_argument("--group-size", type=int, required=True)
    args=parser.parse_args()
    if (args.bits,args.group_size) not in {(3,128),(4,128)}: raise ValueError("RTN accepts only W3/W4 G128")
    if args.backend:
        import subprocess
        backend=resolve_if_rtn_backend(args.backend)
        subprocess.run(["python",str(backend),"--model",str(args.source),"--output",str(args.output),"--wbits",str(args.bits),"--groupsize",str(args.group_size)],check=True)
        provenance={"backend_sha256":sha256_file(backend)}
    else:
        identity=git_identity(args.upstream,GPTQ_URL); sys.path.insert(0,str(args.upstream.resolve()))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from quant import Quantizer, quantize
        model=AutoModelForCausalLM.from_pretrained(args.source,torch_dtype=torch.bfloat16,device_map={"":"cuda:0"},low_cpu_mem_usage=True)
        with torch.no_grad():
            for layer in model.model.layers:
                for module in layer.modules():
                    if not isinstance(module,torch.nn.Linear): continue
                    weight=module.weight.data
                    for start in range(0,weight.shape[1],args.group_size):
                        stop=min(start+args.group_size,weight.shape[1]); block=weight[:,start:stop]
                        q=Quantizer(); q.configure(args.bits,perchannel=True,sym=False,mse=False); q.find_params(block,weight=True)
                        weight[:,start:stop]=quantize(block,q.scale,q.zero,q.maxq).to(weight.dtype)
        args.output.mkdir(parents=True,exist_ok=False); model.save_pretrained(args.output,safe_serialization=True,max_shard_size="5GB"); AutoTokenizer.from_pretrained(args.source).save_pretrained(args.output)
        provenance={"quantizer_repo":identity["origin"],"quantizer_commit":identity["commit"]}
    meta={"quantizer":"RTN","bits":args.bits,"group_size":128,"checkpoint_path":str(args.output),"storage_representation":"HF dequantized quantized-on-grid weights",**provenance}
    (args.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__ == "__main__": main()
