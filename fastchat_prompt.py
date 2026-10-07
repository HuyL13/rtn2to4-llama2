"""Use the official FastChat Vicuna template without importing model adapters."""
from vendor.fastchat_templates.conversation import get_conv_template


def get_conversation_template(model_path):
    # FastChat v0.2.36 VicunaAdapter uses one_shot for v0, otherwise vicuna_v1.1.
    basename = str(model_path).rstrip('/').rsplit('/', 1)[-1]
    if 'vicuna' not in str(model_path).lower():
        raise ValueError('This evaluator only bundles the official Vicuna templates')
    return get_conv_template('one_shot' if 'v0' in basename else 'vicuna_v1.1')
