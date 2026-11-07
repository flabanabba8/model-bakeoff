"""One explicit local model inference at a time through an OpenAI-compatible loopback server (llama.cpp, LM Studio, vLLM) with durable outcome receipts."""
import http.client
import ipaddress
import json
import socket
import time
from urllib.parse import urlsplit
import uuid
from .core import Bakeoff, bounded
from . import locking


def local_endpoint(value, default_port=8080):
    parsed = urlsplit(value)
    host = parsed.hostname
    if host == 'localhost':
        local = True
    else:
        try: local = ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError): local = False
    if parsed.scheme != 'http' or not local or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ValueError('Use a loopback-only HTTP origin for the local model server')
    return host, parsed.port or default_port


class LoopbackConnection(http.client.HTTPConnection):
    """Resolve once and connect only to an address verified as loopback."""
    def connect(self):
        last_error = None
        for family, kind, proto, _, address in socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM):
            try:
                if not ipaddress.ip_address(address[0]).is_loopback: continue
                sock = socket.socket(family, kind, proto)
                try:
                    sock.settimeout(self.timeout)
                    sock.connect(address)
                except OSError:
                    sock.close()
                    raise
                self.sock = sock
                return
            except (OSError, ValueError) as exc:
                last_error = exc
        raise OSError('No reachable loopback address') from last_error


def execute(bakeoff, bakeoff_id, task_id, model, base_url=None, seed=0, timeout_seconds=120, max_tokens=2048):
    path = bakeoff.path(bakeoff_id)  # Validate before using the ID in a lock filename.
    if not path.is_file(): raise ValueError('Unknown bakeoff')
    with locking.exclusive_lock(bakeoff.root / (bakeoff_id + '.run.lock'), 'Another local model run is active for this bakeoff', error=ValueError):
        return _execute(bakeoff, bakeoff_id, task_id, model, base_url, seed, timeout_seconds, max_tokens)


def _execute(bakeoff, bakeoff_id, task_id, model, base_url, seed, timeout_seconds, max_tokens):
    bounded(model, 'model label', 200)
    bounded(task_id, 'task ID', 100)
    if type(seed) is not int or not 0 <= seed <= 2**31-1: raise ValueError('Seed must be a nonnegative 32-bit integer')
    if type(timeout_seconds) not in (int, float) or not 10 <= timeout_seconds <= 3600: raise ValueError('timeout_seconds must be 10 to 3600')
    if type(max_tokens) is not int or not 16 <= max_tokens <= 32768: raise ValueError('max_tokens must be 16 to 32768')
    if base_url is None: base_url = 'http://localhost:8080'
    host, port = local_endpoint(base_url)
    path = bakeoff.path(bakeoff_id)
    with locking.exclusive_lock(bakeoff.root / '.lock', 'Benchmark state is busy', blocking=True):
        data = bakeoff.load(path)
        task = next((x for x in data['tasks'] if x['id'] == task_id), None)
        if task is None: raise ValueError('Unknown task')
        if task_id in data['results'].get(model, {}): raise ValueError('Result already recorded for this model and task')
        attempts = data.setdefault('attempts', [])
        if any(x['status'] == 'pending' for x in attempts):
            raise ValueError('Another model run is still pending for this bakeoff')
        if any(x['model'] == model and x['task_id'] == task_id and x['status'] in ('pending', 'unknown') for x in attempts):
            raise ValueError('A prior run has an unresolved outcome; do not automatically retry')
        attempt = {'id': uuid.uuid4().hex, 'model': model, 'task_id': task_id, 'status': 'pending', 'seed': seed, 'lock_version': 1}
        attempts.append(attempt)
        bakeoff.save(path, data)
    started = time.monotonic()
    try:
        connection = LoopbackConnection(host, port, timeout=timeout_seconds)
        try:
            payload = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': task['prompt']}], 'stream': False,
                                  'temperature': 0, 'seed': seed, 'max_tokens': max_tokens}).encode()
            endpoint = '/v1/chat/completions'
            connection.request('POST', endpoint, body=payload, headers={'Content-Type': 'application/json', 'Connection': 'close'})
            response = connection.getresponse()
            if response.status != 200: raise ValueError('Server did not accept the request')
            raw = response.read(4 * 1024**2 + 1)
            if len(raw) > 4 * 1024**2: raise ValueError('Response exceeds limit')
            result = json.loads(raw)
        finally:
            connection.close()
        choice = result['choices'][0] if isinstance(result, dict) and isinstance(result.get('choices'), list) and result['choices'] else None
        if not isinstance(choice, dict) or choice.get('finish_reason') != 'stop' or not isinstance(choice.get('message'), dict) or not isinstance(choice['message'].get('content'), str) or len(choice['message']['content']) > 100000:
            raise ValueError('Incomplete model response')
        answer, keys = choice['message']['content'], ('prompt_tokens', 'completion_tokens')
        container = result.get('usage') if isinstance(result.get('usage'), dict) else {}
        result_model = result.get('model', '')
        reported = []
        for key in keys:
            value = container.get(key)
            reported.append(value if type(value) is int and 0 <= value <= 10**12 else None)
        elapsed = round((time.monotonic() - started) * 1000)
    except (OSError, ValueError, TypeError, TimeoutError, http.client.HTTPException):
        with locking.exclusive_lock(bakeoff.root / '.lock', 'Benchmark state is busy', blocking=True):
            data = bakeoff.load(path)
            next(x for x in data['attempts'] if x['id'] == attempt['id'])['status'] = 'unknown'
            bakeoff.save(path, data)
        raise RuntimeError('Local model request failed or its outcome is unknown; no automatic retry was made') from None
    with locking.exclusive_lock(bakeoff.root / '.lock', 'Benchmark state is busy', blocking=True):
        data = bakeoff.load(path)
        stored = next(x for x in data['attempts'] if x['id'] == attempt['id'])
        if task_id in data['results'].get(model, {}):
            stored.update({'status': 'completed_conflict', 'answer': answer})
            bakeoff.save(path, data)
            raise ValueError('A result was recorded concurrently; generated answer was retained in the private attempt')
        passed = answer == task['expected'] if task['check'] == 'exact' else task['expected'] in answer
        data['results'].setdefault(model, {})[task_id] = {'answer': answer, 'passed': passed, 'latency_ms': elapsed,
            'input_tokens': reported[0], 'output_tokens': reported[1], 'origin': 'local_openai_compatible',
            'reported_model': str(result_model)[:200], 'seed': seed}
        stored['status'] = 'complete'
        bakeoff.save(path, data)
    return {'model': model, 'task_id': task_id, 'passed': passed, 'latency_ms': elapsed,
            'input_tokens': reported[0], 'output_tokens': reported[1], 'provider_reported_model': str(result_model)[:200],
            'notice': 'One local generation was performed through an OpenAI-compatible local server. Matching a literal check does not establish general quality.'}
