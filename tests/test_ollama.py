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
spec = importlib.util.spec_from_file_location('bakeoff_ollama_fixture', root / '__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class Fixture(http.server.BaseHTTPRequestHandler):
    requests = 0
    def do_POST(self):
        type(self).requests += 1
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if request['prompt'] == 'Fail safely':
            self.close_connection = True
            return
        data = json.dumps({'model': request['model'], 'response': 'yes', 'done': True,
                           'prompt_eval_count': 5, 'eval_count': 1}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args): pass

class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.plugin = module.Bakeoff(Path(self.tmp.name))
        self.id = self.plugin.create([{'id': 'a', 'prompt': 'Say yes', 'check': 'exact', 'expected': 'yes'},
                                      {'id': 'b', 'prompt': 'Fail safely', 'check': 'exact', 'expected': 'yes'}])['bakeoff_id']
        Fixture.requests = 0
        self.server = http.server.HTTPServer(('localhost', 0), Fixture)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://localhost:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self.tmp.cleanup()

    def test_actual_local_http_result_and_usage(self):
        result = module.execute(self.plugin, self.id, 'a', 'fixture-model', base_url=self.url, seed=7)
        self.assertTrue(result['passed'])
        self.assertEqual((result['input_tokens'], result['output_tokens']), (5, 1))
        report = self.plugin.report(self.id)
        self.assertEqual(report['models'][0]['passed'], 1)
        self.assertEqual(report['attempts'][0]['status'], 'complete')
        self.assertEqual(Fixture.requests, 1)

    def test_unknown_outcome_blocks_automatic_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            module.execute(self.plugin, self.id, 'b', 'fixture-model', base_url=self.url)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            module.execute(self.plugin, self.id, 'b', 'fixture-model', base_url=self.url)
        self.assertEqual(Fixture.requests, 1)
        self.assertEqual(self.plugin.report(self.id)['attempts'][0]['status'], 'unknown')
        self.plugin.abandon(self.id, 'b', 'fixture-model', 'Reviewed the interrupted fixture')
        self.assertEqual(Fixture.requests, 1)
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            module.execute(self.plugin, self.id, 'b', 'fixture-model', base_url=self.url)
        self.assertEqual(Fixture.requests, 2)

    def test_pending_run_blocks_parallel_work(self):
        path = self.plugin.path(self.id)
        data = self.plugin.load(path)
        data['attempts'].append({'id':'fixture', 'model':'first', 'task_id':'a', 'status':'pending', 'seed':0})
        self.plugin.save(path, data)
        with self.assertRaisesRegex(ValueError, 'still pending'):
            module.execute(self.plugin, self.id, 'b', 'second', base_url=self.url)
        with self.assertRaisesRegex(ValueError, 'older records'):
            self.plugin.recover_pending(self.id)
        self.assertEqual(Fixture.requests, 0)

    def test_hostname_resolution_cannot_connect_to_remote_address(self):
        remote = str(ipaddress.IPv4Address(0x08080808))
        address = [(module.ollama.socket.AF_INET, module.ollama.socket.SOCK_STREAM, 6, '', (remote, 11434))]
        with patch.object(module.ollama.socket, 'getaddrinfo', return_value=address), patch.object(module.ollama.socket, 'socket') as create:
            with self.assertRaisesRegex(OSError, 'loopback'):
                module.ollama.LoopbackConnection('localhost', 11434, timeout=1).connect()
        create.assert_not_called()

    def test_remote_endpoint_rejected_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, 'loopback'):
            module.execute(self.plugin, self.id, 'a', 'fixture-model', base_url='http://example.com')
        self.assertEqual(Fixture.requests, 0)

if __name__ == '__main__': unittest.main()
