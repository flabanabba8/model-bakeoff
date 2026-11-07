import http.server
import importlib.util
import ipaddress
from unittest.mock import patch
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('bakeoff_local_fixture', root / '__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Fixture(http.server.BaseHTTPRequestHandler):
    """A minimal OpenAI-compatible chat server such as llama.cpp's."""
    requests = []

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).requests.append((self.path, request))
        prompt = request['messages'][0]['content']
        if prompt == 'Fail safely':
            self.close_connection = True
            return
        reason = 'length' if prompt == 'Run too long' else 'stop'
        data = json.dumps({'model': request['model'], 'choices': [{'index': 0, 'finish_reason': reason,
                           'message': {'role': 'assistant', 'content': 'yes', 'reasoning_content': 'thinking about it'}}],
                           'usage': {'prompt_tokens': 9, 'completion_tokens': 3}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args): pass


class LocalServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.plugin = module.Bakeoff(Path(self.tmp.name))
        self.id = self.plugin.create([{'id': 'a', 'prompt': 'Say yes', 'check': 'exact', 'expected': 'yes'},
                                      {'id': 'b', 'prompt': 'Fail safely', 'check': 'exact', 'expected': 'yes'},
                                      {'id': 'c', 'prompt': 'Run too long', 'check': 'exact', 'expected': 'yes'}])['bakeoff_id']
        Fixture.requests = []
        self.server = http.server.HTTPServer(('localhost', 0), Fixture)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://localhost:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self.tmp.cleanup()

    def test_result_usage_request_shape_and_final_answer_only(self):
        result = module.execute(self.plugin, self.id, 'a', 'fixture-model', base_url=self.url, seed=7)
        self.assertTrue(result['passed'])
        self.assertEqual((result['input_tokens'], result['output_tokens']), (9, 3))
        path, request = Fixture.requests[0]
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual((request['temperature'], request['seed'], request['stream']), (0, 7, False))
        self.assertEqual(self.plugin.report(self.id)['models'][0]['passed'], 1)
        self.assertEqual(self.plugin.report(self.id)['attempts'][0]['status'], 'complete')
        self.assertNotIn('thinking', json.dumps(self.plugin.result(self.id, 'a', 'fixture-model')))

    def test_unknown_outcome_blocks_automatic_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            module.execute(self.plugin, self.id, 'b', 'fixture-model', base_url=self.url)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            module.execute(self.plugin, self.id, 'b', 'fixture-model', base_url=self.url)
        self.assertEqual(len(Fixture.requests), 1)
        self.assertEqual(self.plugin.report(self.id)['attempts'][0]['status'], 'unknown')
        self.plugin.abandon(self.id, 'b', 'fixture-model', 'Reviewed the interrupted fixture')
        self.assertEqual(len(Fixture.requests), 1)
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            module.execute(self.plugin, self.id, 'b', 'fixture-model', base_url=self.url)
        self.assertEqual(len(Fixture.requests), 2)

    def test_truncated_generation_is_unknown_and_never_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            module.execute(self.plugin, self.id, 'c', 'fixture-model', base_url=self.url)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            module.execute(self.plugin, self.id, 'c', 'fixture-model', base_url=self.url)
        self.assertEqual(len(Fixture.requests), 1)

    def test_pending_run_blocks_parallel_work(self):
        path = self.plugin.path(self.id)
        data = self.plugin.load(path)
        data['attempts'].append({'id': 'fixture', 'model': 'first', 'task_id': 'a', 'status': 'pending', 'seed': 0})
        self.plugin.save(path, data)
        with self.assertRaisesRegex(ValueError, 'still pending'):
            module.execute(self.plugin, self.id, 'b', 'second', base_url=self.url)
        with self.assertRaisesRegex(ValueError, 'older records'):
            self.plugin.recover_pending(self.id)
        self.assertEqual(Fixture.requests, [])

    def test_live_execution_holds_recovery_lock(self):
        entered, release, outcomes = threading.Event(), threading.Event(), []
        def work(*args):
            entered.set()
            if not release.wait(5): raise RuntimeError('fixture timed out')
            return {'status': 'fixture complete'}
        with patch.object(module.local, '_execute', side_effect=work):
            thread = threading.Thread(target=lambda: outcomes.append(module.execute(self.plugin, self.id, 'a', 'fixture-model', base_url=self.url)))
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(ValueError, 'still active'):
                    self.plugin.recover_pending(self.id)
            finally:
                release.set(); thread.join(timeout=5)
        self.assertEqual(outcomes, [{'status': 'fixture complete'}])

    def test_hostname_resolution_cannot_connect_to_remote_address(self):
        remote = str(ipaddress.IPv4Address(0x08080808))
        address = [(module.local.socket.AF_INET, module.local.socket.SOCK_STREAM, 6, '', (remote, 8080))]
        with patch.object(module.local.socket, 'getaddrinfo', return_value=address), patch.object(module.local.socket, 'socket') as create:
            with self.assertRaisesRegex(OSError, 'loopback'):
                module.local.LoopbackConnection('localhost', 8080, timeout=1).connect()
        create.assert_not_called()

    def test_remote_endpoint_and_limits_rejected_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, 'loopback'):
            module.execute(self.plugin, self.id, 'a', 'fixture-model', base_url='http://example.com')
        for kwargs in ({'timeout_seconds': 5}, {'max_tokens': 4}, {'seed': -1}):
            with self.assertRaises(ValueError): module.execute(self.plugin, self.id, 'a', 'm', base_url=self.url, **kwargs)
        self.assertEqual(Fixture.requests, [])


if __name__ == '__main__': unittest.main()
