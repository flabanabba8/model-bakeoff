"""One explicit local Ollama inference at a time with durable outcome receipts."""
import fcntl
import http.client
import ipaddress
import json
import socket
import time
from urllib.parse import urlsplit
import uuid
from .core import Bakeoff, bounded


def local_endpoint(value):
    parsed = urlsplit(value)
    host = parsed.hostname
    if host == 'localhost':
        local = True
    else:
        try: local = ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError): local = False
    if parsed.scheme != 'http' or not local or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ValueError('Use a loopback-only HTTP Ollama origin')
    return host, parsed.port or 11434


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


def execute(bakeoff, bakeoff_id, task_id, model, base_url='http://localhost:11434', seed=0):
    bounded(model, 'model label', 200)
    bounded(task_id, 'task ID', 100)
    if type(seed) is not int or not 0 <= seed <= 2**31-1: raise ValueError('Seed must be a nonnegative 32-bit integer')
    host, port = local_endpoint(base_url)
    path = bakeoff.path(bakeoff_id)
    with (bakeoff.root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = bakeoff.load(path)
        task = next((x for x in data['tasks'] if x['id'] == task_id), None)
        if task is None: raise ValueError('Unknown task')
        if task_id in data['results'].get(model, {}): raise ValueError('Result already recorded for this model and task')
        attempts = data.setdefault('attempts', [])
        if any(x['status'] == 'pending' for x in attempts):
            raise ValueError('Another model run is still pending for this bakeoff')
        if any(x['model'] == model and x['task_id'] == task_id and x['status'] in ('pending', 'unknown') for x in attempts):
            raise ValueError('A prior run has an unresolved outcome; do not automatically retry')
        attempt = {'id': uuid.uuid4().hex, 'model': model, 'task_id': task_id, 'status': 'pending', 'seed': seed}
        attempts.append(attempt)
        bakeoff.save(path, data)
    started = time.monotonic()
    try:
        connection = LoopbackConnection(host, port, timeout=120)
        try:
            payload = json.dumps({'model': model, 'prompt': task['prompt'], 'stream': False,
                                  'options': {'seed': seed, 'temperature': 0}}).encode()
            connection.request('POST', '/api/generate', body=payload, headers={'Content-Type': 'application/json', 'Connection': 'close'})
            response = connection.getresponse()
            if response.status != 200: raise ValueError('Server did not accept the request')
            raw = response.read(4 * 1024**2 + 1)
            if len(raw) > 4 * 1024**2: raise ValueError('Response exceeds limit')
            result = json.loads(raw)
        finally:
            connection.close()
        if not isinstance(result, dict) or result.get('done') is not True or not isinstance(result.get('response'), str) or len(result['response']) > 100000:
            raise ValueError('Incomplete model response')
        reported = []
        for key in ('prompt_eval_count', 'eval_count'):
            value = result.get(key)
            reported.append(value if type(value) is int and 0 <= value <= 10**12 else None)
        elapsed = round((time.monotonic() - started) * 1000)
    except (OSError, ValueError, TypeError, TimeoutError, http.client.HTTPException):
        with (bakeoff.root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = bakeoff.load(path)
            next(x for x in data['attempts'] if x['id'] == attempt['id'])['status'] = 'unknown'
            bakeoff.save(path, data)
        raise RuntimeError('Local model request failed or its outcome is unknown; no automatic retry was made') from None
    with (bakeoff.root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = bakeoff.load(path)
        stored = next(x for x in data['attempts'] if x['id'] == attempt['id'])
        if task_id in data['results'].get(model, {}):
            stored.update({'status': 'completed_conflict', 'answer': result['response']})
            bakeoff.save(path, data)
            raise ValueError('A result was recorded concurrently; generated answer was retained in the private attempt')
        passed = result['response'] == task['expected'] if task['check'] == 'exact' else task['expected'] in result['response']
        data['results'].setdefault(model, {})[task_id] = {'answer': result['response'], 'passed': passed, 'latency_ms': elapsed,
            'input_tokens': reported[0], 'output_tokens': reported[1], 'origin': 'local_ollama',
            'reported_model': str(result.get('model', ''))[:200], 'seed': seed}
        stored['status'] = 'complete'
        bakeoff.save(path, data)
    return {'model': model, 'task_id': task_id, 'passed': passed, 'latency_ms': elapsed,
            'input_tokens': reported[0], 'output_tokens': reported[1], 'provider_reported_model': str(result.get('model', ''))[:200],
            'notice': 'One local generation was performed. Matching a literal check does not establish general quality.'}
