"""Immutable probability contract and Brier scoring, shared by all graders."""
import json
import math


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('confidence must be a numeric probability')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('confidence must be finite and in [0,1]')
    return value


def parse_model_probability(text):
    """Read the final bare probability line after optional verification steps."""
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError('Missing final probability line')
    value = json.loads(lines[-1].strip())
    return probability(value)


def score(ids, outputs, labels):
    details = []
    valid = 0
    for eid in ids:
        p = None
        try:
            p = probability(outputs[eid]['confidence'])
            valid += 1
            loss = (p - labels[eid]) ** 2
        except (KeyError, ValueError, TypeError):
            loss = 1.0
        details.append({'example_id': eid, 'gold_label': labels[eid],
                        'confidence': p, 'brier_loss': loss})
    return {'n': len(ids), 'brier_score': sum(r['brier_loss'] for r in details) / len(ids),
            'valid': valid, 'invalid': len(ids) - valid,
            'constant_half_brier': 0.25, 'per_instance': details}


def load_outputs(path, ids):
    from pathlib import Path
    allowed, outputs = set(ids), {}
    if not Path(path).exists():
        return outputs
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
            eid = row['example_id']
            if eid in allowed and eid not in outputs:
                outputs[eid] = row
        except (ValueError, KeyError, TypeError):
            pass
    return outputs
