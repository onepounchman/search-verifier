"""Merge a pure text-model LoRA into the original multimodal checkpoint layout."""
import argparse
import json
import shutil
from pathlib import Path
from paths import MODEL_PATH, configure_runtime, validate_model_output


def main():
    configure_runtime()
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cli = argparse.ArgumentParser()
    cli.add_argument('--adapter', required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--model-path', default=str(MODEL_PATH))
    args = cli.parse_args()
    validate_model_output(args.output)
    validate_model_output(args.adapter)
    adapter = Path(args.adapter)
    config = json.loads((adapter / 'adapter_config.json').read_text())
    assert config.get('peft_type') == 'LORA'
    assert not config.get('modules_to_save') and config.get('bias', 'none') == 'none'
    tensors = load_file(str(adapter / 'adapter_model.safetensors'))
    assert all('lora_A' in name or 'lora_B' in name for name in tensors), 'Non-LoRA adapter parameters'
    updated_weight_keys = set()
    for name in tensors:
        module = name.split('.lora_A', 1)[0].split('.lora_B', 1)[0]
        assert module.startswith('base_model.model.'), 'Unexpected LoRA module path'
        updated_weight_keys.add(module[len('base_model.model.'):] + '.weight')
    base = Path(args.model_path)
    model = Qwen3_5ForCausalLM.from_pretrained(base,
        config=Qwen3_5TextConfig.from_pretrained(base, local_files_only=True),
        dtype=torch.bfloat16, local_files_only=True)
    merged = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    original_index = json.loads((base / 'model.safetensors.index.json').read_text())
    original = {}
    for name in set(original_index['weight_map'].values()):
        original.update(load_file(str(base / name)))
    updated = 0
    for key, tensor in merged.state_dict().items():
        if key not in updated_weight_keys:
            # Preserve untargeted parameters, including FP32 GDN normalization.
            continue
        mapped = 'model.language_model.' + key[len('model.'):] if key.startswith('model.') else key
        if mapped not in original and key in original:
            mapped = key
        if mapped not in original and key == 'lm_head.weight' and model.config.tie_word_embeddings:
            # The source checkpoint serializes the tied embeddings only once.
            continue
        assert mapped in original, 'Cannot map text weight: ' + key
        original[mapped] = tensor.detach().cpu().to(original[mapped].dtype).contiguous()
        updated += 1
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for path in base.iterdir():
        if path.is_file() and not path.name.endswith('.safetensors') and path.name != 'model.safetensors.index.json':
            shutil.copyfile(path, out / path.name)
    save_file(original, str(out / 'model.safetensors'), metadata={'format': 'pt'})
    (out / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': sum(t.numel() * t.element_size() for t in original.values())},
        'weight_map': {k: 'model.safetensors' for k in original}}, indent=2))
    print(json.dumps({'merged_text_tensors': updated, 'output': str(out)}))


if __name__ == '__main__':
    main()
