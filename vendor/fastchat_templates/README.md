This directory bundles unmodified `fastchat/conversation.py` and the Apache-2.0
LICENSE from lm-sys/FastChat v0.2.36:
https://github.com/lm-sys/FastChat/tree/v0.2.36

Only the official conversation registry is imported. No model adapters, Torch,
Transformers patches, or optional inference engines are imported by this module.
`fastchat_prompt.py` applies the official VicunaAdapter template selection:
`vicuna` -> `vicuna_v1.1`, paths containing `v0` -> `one_shot`.
Prompt rendering and scoring are unchanged.
