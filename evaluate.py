"""Local Brier evaluation. Gold labels are read only by this scoring process."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from probability_scoring import load_outputs, score
from paths import configure_runtime


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--inputs', required=True)
    cli.add_argument('--labels', required=True)
    cli.add_argument('--predictions', help='Score saved outputs without another model call')
    cli.add_argument('--run-dir', required=True)
    cli.add_argument('--serve-python', default=os.environ.get('PILOT_SERVE_PYTHON', sys.executable))
    cli.add_argument('--judge', default=str(Path(__file__).with_name('judge.py')))
    cli.add_argument('--time-limit', type=int, default=7200)
    args = cli.parse_args()
    assert args.time_limit > 0
    ids = [r['example_id'] for r in map(json.loads, Path(args.inputs).open())]
    assert ids and len(ids) == len(set(ids))
    labels = {r['example_id']: r['gold_label'] for r in map(json.loads, Path(args.labels).open())}
    assert set(ids) <= labels.keys() and all(labels[eid] in (0, 1) for eid in ids)
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    predictions = Path(args.predictions) if args.predictions else run / 'outputs.jsonl'
    started = time.monotonic()
    timed_out, rc = False, None
    if not args.predictions:
        configure_runtime()
        with (run / 'method.log').open('w') as log:
            process = subprocess.Popen([args.serve_python, args.judge, '--inputs', str(Path(args.inputs).resolve()),
                '--outputs', str(predictions)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = process.wait(timeout=args.time_limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                import psutil
                try:
                    parent = psutil.Process(process.pid)
                    owned = parent.children(recursive=True) + [parent]
                    for item in reversed(owned):
                        try: item.terminate()
                        except psutil.NoSuchProcess: pass
                    _, alive = psutil.wait_procs(owned, timeout=3)
                    for item in alive:
                        try: item.kill()
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess:
                    pass
                process.wait()
                rc = process.returncode
    metrics = score(ids, load_outputs(predictions, ids), labels)
    detail = metrics.pop('per_instance')
    (run / 'per_instance.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in detail))
    metrics.update(wall_seconds=time.monotonic()-started, timed_out=timed_out, exit_code=rc,
                   predictions_reused=bool(args.predictions), time_limit_seconds=args.time_limit)
    (run / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
