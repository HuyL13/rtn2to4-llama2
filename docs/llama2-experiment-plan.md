# Llama-2-7B experiment

All newly trained source models use meta-llama/Llama-2-7b-hf. IF-SFT uses the existing cnut1648/LLaMA2-7B-fingerprinted-SFT checkpoint. Generation helpers also use Llama-2-7B (chat for natural-language query construction).

1. Vendor the requested local RBVT lm-eval runner and the newest local ImF native ADG implementation; record provenance. Adapt only the copied ImF backbone pin and training compatibility.
2. Reuse SewoongLab generation/full fine-tuning/evaluation; retain the actual post-tokenization training pairs for evaluation. Generate Perinucleus targets on Llama-2. Reuse CTCC datasets, training recipe, LLaMA-Factory, and prompt construction, with paper exact-match scoring.
3. Add runtime audit of original RTN2 assignments, original cell boundaries and allowed four fine levels after dtype conversion. Abort on any violation. Preserve the existing quantizer and selected modules.
4. Evaluate source, RTN2, direct RTN4, and nested 2-to-4 independently from the same source; reuse existing PPL; run local copied lm-eval on ARC-Challenge and ARC-Easy only. Use a temporary dense HF export so quantized in-memory weights are actually evaluated.
5. Provide a server runner that trains missing checkpoints, gates clean fingerprint success, resumes completed outputs, and never installs dependencies, activates environments or clones repositories.
6. Verify CPU contracts, copied native codec tests, syntax, and bash syntax. GPU training and benchmark numbers are executed by the user on their server.

Native metrics: IF upstream FSR; English-Random/Perinucleus greedy one-token exact recall; ImF decoded-payload success with clean-base false verification; CTCC exact target response on triggers and activation on negative inputs. NLL remains diagnostic.

ImF is the local reconstructed ADG experiment, not a recovery of the released missing decoder/key. Its exact-reference query constructor and lack of full paper iterative/manual selection are documented, not presented as an exact paper reproduction.
