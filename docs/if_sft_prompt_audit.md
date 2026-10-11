# Compare native inference roles with training roles

The original IF inference appends conversation roles literally (`human`), while
the bundled official training preprocessing maps `human` to Vicuna's `USER`.
`scripts/if_sft_fidelity.py` reuses the existing audit and scorer to compare
these two prompts on the same eight fingerprint examples. Both prompts retain
the same system message and `ASSISTANT: Based on my fingerprint, the message is:`
prefix. This tests role formatting, not Llama2's `[INST]` format: this checkpoint
was trained using Vicuna formatting.

Run the same comparison on the new Adam8 source and that source with the MEraser
erase adapter. A 128-token budget permits checking whether a second incomplete
fingerprint was merely cut off at the former 30-token limit. Raw generated IDs,
token count and EOS/budget finish reason are saved, along with each prompt and
answer. ID 2's prompt and answer print directly to the cell.

These exact/contains percentages are prompt diagnostics with a different token
budget; they do not replace or retroactively change the native FSR reports.
Neither training nor erasure is rerun. The native evaluator is unchanged.

```python
%cd /content/rtn2to4-llama2
!git pull --ff-only
!python -u scripts/if_sft_fidelity.py --model-path outputs/original_if_tfa_adam8_v1/checkpoint --split validation --max-new-tokens 128 --output-dir outputs/original_if_tfa_adam8_v1/prompt_audit_128/base
!python -u scripts/if_sft_fidelity.py --model-path outputs/original_if_tfa_adam8_v1/checkpoint --adapter-path outputs/original_if_tfa_adam8_v1/meraser_erase_v1/erase_adapter --split validation --max-new-tokens 128 --output-dir outputs/original_if_tfa_adam8_v1/prompt_audit_128/erase
```

Local verification covers prompt construction and CLI/scorer wiring, not
execution of the user's GPU models. The two inference commands above provide
the actual comparison on those checkpoints.
