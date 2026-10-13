"""Weak TRL LoRA starter; improve its data construction and hyperparameters."""
import argparse
import json
import os
from pathlib import Path
from paths import MODEL_PATH, configure_runtime, validate_model_output


def main():
    configure_runtime()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from trl import SFTConfig, SFTTrainer
    cli = argparse.ArgumentParser()
    cli.add_argument('--train-ids', required=True)
    cli.add_argument('--output-dir', required=True)
    cli.add_argument('--max-steps', type=int, default=32)
    cli.add_argument('--max-length', type=int, default=8192)
    cli.add_argument('--gradient-accumulation', type=int, default=8)
    cli.add_argument('--learning-rate', type=float, default=1e-4)
    cli.add_argument('--rank', type=int, default=16)
    cli.add_argument('--train-inputs', required=True)
    cli.add_argument('--train-labels', required=True)
    cli.add_argument('--model-path', default=str(MODEL_PATH))
    args = cli.parse_args()
    out = Path(args.output_dir).resolve()
    validate_model_output(out)
    ids = set(json.loads(Path(args.train_ids).read_text()))
    labels = {r['example_id']: r['gold_label'] for r in map(json.loads, Path(args.train_labels).open())}
    assert ids <= set(labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    data = []
    for line in Path(args.train_inputs).open():
        row = json.loads(line)
        if row['example_id'] not in ids:
            continue
        record = row['input']
        prompt = [{'role': 'system', 'content': 'Verify the recorded final answer step by step. End with a final line containing only its probability of being correct, a number between 0 and 1 such as 0.75, without braces or a label.'},
                  {'role': 'user', 'content': json.dumps({'question': record['question'], 'trajectory': record['messages']}, ensure_ascii=False)}]
        prefix = tokenizer.apply_chat_template(prompt, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        target = tokenizer.encode(str(float(labels[row['example_id']])), add_special_tokens=False) + [tokenizer.eos_token_id]
        budget = args.max_length - len(target)
        if len(prefix) > budget:
            head = min(1024, budget // 3)
            prefix = prefix[:head] + prefix[-(budget - head):]
        data.append({'input_ids': prefix + target, 'labels': [-100] * len(prefix) + target})
    torch.manual_seed(42)
    config = Qwen3_5TextConfig.from_pretrained(args.model_path, local_files_only=True)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(args.model_path, config=config,
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    train = SFTConfig(output_dir=str(out), max_length=args.max_length,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps, learning_rate=args.learning_rate, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        completion_only_loss=True, dataset_kwargs={'skip_prepare_dataset': True},
        packing=False, padding_free=False, logging_steps=1, save_strategy='steps',
        save_steps=args.max_steps, save_total_limit=1, report_to='none')
    trainer = SFTTrainer(model=model, args=train, train_dataset=Dataset.from_list(data),
        processing_class=tokenizer, peft_config=LoraConfig(task_type='CAUSAL_LM',
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0,
            target_modules='all-linear', bias='none'))
    assert all('lora_' in name for name, p in trainer.model.named_parameters() if p.requires_grad)
    trainer.train()
    trainer.model.save_pretrained(out / 'adapter', safe_serialization=True)
    (out / 'train_ids.json').write_text(json.dumps(sorted(ids)))


if __name__ == '__main__':
    main()
