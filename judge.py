"""Single-pass Qwen judge; replace its pipeline while preserving the CLI contract."""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from probability_scoring import parse_model_probability
from paths import MODEL_PATH, configure_runtime, validate_model_output


def messages(row, system):
    payload = row['input']
    record = {'question': payload['question'], 'recorded_trajectory': payload['messages'],
              'available_tool_definitions': payload['tool_registry']['tools']}
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': 'Verify this recorded search run:\n' +
             json.dumps(record, ensure_ascii=False, separators=(',', ':'))}]


async def generate(records, config, tokenizer, destination):
    import aiohttp
    settings = config['sampling_params']
    semaphore = asyncio.Semaphore(config.get('concurrency', 32))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as session:
        with destination.open('w') as handle:
            async def one(row):
                async with semaphore:
                    started = time.monotonic()
                    ids = tokenizer.apply_chat_template(messages(row, config['system_prompt']),
                        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=config.get('thinking', False))
                    if len(ids) + settings['max_new_tokens'] > 262144:
                        raise ValueError('Prompt exceeds native context; redesign input preprocessing')
                    async with session.post('http://127.0.0.1:31000/generate',
                        json={'input_ids': ids, 'sampling_params': settings}) as response:
                        response.raise_for_status()
                        answer = await response.json()
                    try:
                        confidence = parse_model_probability(answer['text'])
                    except (ValueError, TypeError):
                        confidence = None
                    output = {'example_id': row['example_id'], 'confidence': confidence, 'raw_text': answer['text'],
                              'request_seconds': time.monotonic() - started,
                              'meta_info': answer.get('meta_info', {})}
                    handle.write(json.dumps(output, ensure_ascii=False) + '\n')
                    handle.flush()
            results = await asyncio.gather(*(one(r) for r in records), return_exceptions=True)
            for row, result in zip(records, results):
                if isinstance(result, Exception):
                    handle.write(json.dumps({'example_id': row['example_id'], 'confidence': None, 'error': str(result)}) + '\n')
                    handle.flush()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--inputs', required=True)
    cli.add_argument('--outputs', required=True)
    cli.add_argument('--model-path', default=str(MODEL_PATH))
    args = cli.parse_args()
    configure_runtime()
    here = Path(__file__).resolve().parent
    config = json.loads((here / 'config.json').read_text())
    os.environ['PYTHONPATH'] = str(here) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('USER', 'pilot'); os.environ.setdefault('LOGNAME', 'pilot')
    records = list(map(json.loads, Path(args.inputs).open()))
    output = Path(args.outputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = str(Path(args.model_path).resolve())
    os.environ['RSI_BASE_MODEL_PATH'] = model
    adapter = config.get('adapter_path')
    if adapter:
        # The trusted merger only accepts LoRA adapter files and the fixed base.
        merge_root = Path(os.environ['PILOT_OUTPUT_ROOT']) / 'serving_merges'
        merge_root.mkdir(parents=True, exist_ok=True)
        model = str(merge_root / ('probability_' + str(os.getpid()) + '_' + str(time.time_ns())))
        validate_model_output(adapter)
        subprocess.run([os.environ['PILOT_TRAIN_PYTHON'], str(here / 'merge_adapter.py'),
            '--adapter', adapter, '--output', model], check=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    server_args = config.get('server_args', [])
    log = (output.parent / 'server.log').open('w')
    server = subprocess.Popen([sys.executable, '-m', 'sglang.launch_server',
        '--model-path', model, '--host', '127.0.0.1', '--port', '31000',
        '--dp-size', '1', '--tp-size', '1', '--dtype', 'bfloat16',
        '--context-length', '262144', '--mem-fraction-static', '0.90',
        '--max-running-requests', '32', '--max-mamba-cache-size', '64',
        '--mamba-radix-cache-strategy', 'extra_buffer', '--disable-radix-cache',
        '--chunked-prefill-size', '8192', '--attention-backend', 'flashinfer',
        '--linear-attn-backend', 'triton', '--cuda-graph-max-bs-decode', '32',
        '--random-seed', '20261010', *server_args],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError('Serving startup failed; inspect server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:31000/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError('Serving startup timeout')
        asyncio.run(generate(records, config, tokenizer, output))
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()


if __name__ == '__main__':
    main()
