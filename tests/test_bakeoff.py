import importlib.util
import json
import fcntl
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('bakeoff_core', Path(__file__).parents[1] / 'core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

class BakeoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plugin = core.Bakeoff(self.root)
        self.tasks = [{'id': 'a', 'prompt': 'Say yes', 'check': 'exact', 'expected': 'yes'}, {'id': 'b', 'prompt': 'Say hello', 'check': 'contains', 'expected': 'hello'}]
        self.id = self.plugin.create(self.tasks)['bakeoff_id']

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_and_report_missing(self):
        self.plugin.record(self.id, 'a', 'model-a', 'yes', latency_ms=100, input_tokens=3, output_tokens=1)
        self.plugin.record(self.id, 'b', 'model-a', 'hello there')
        self.plugin.record(self.id, 'a', 'model-b', 'no')
        report = self.plugin.report(self.id)['models']
        self.assertEqual((report[0]['passed'], report[0]['recorded'], report[0]['latency_samples']), (2, 2, 1))
        self.assertEqual(report[1]['missing_task_ids'], ['b'])
        with self.assertRaisesRegex(ValueError, 'already'):
            self.plugin.record(self.id, 'a', 'model-a', 'yes')

    def test_paired_comparison_excludes_unshared_tasks(self):
        self.plugin.record(self.id, 'a', 'model-a', 'yes')
        self.plugin.record(self.id, 'b', 'model-a', 'hello')
        self.plugin.record(self.id, 'a', 'model-b', 'no')
        result = self.plugin.compare(self.id, 'model-a', 'model-b')
        self.assertEqual(result['shared_task_ids'], ['a'])
        self.assertEqual(result['a_only_task_ids'], ['b'])
        self.assertEqual((result['only_a_passed'], result['only_b_passed']), (1, 0))
        with self.assertRaises(ValueError): self.plugin.compare(self.id, 'model-a', 'model-a')
        with self.assertRaises(ValueError): self.plugin.compare(self.id, 'model-a', 'unknown')

    def test_no_shared_tasks_are_not_comparable(self):
        self.plugin.record(self.id, 'a', 'model-a', 'yes')
        self.plugin.record(self.id, 'b', 'model-b', 'hello')
        result = self.plugin.compare(self.id, 'model-a', 'model-b')
        self.assertEqual(result['status'], 'no_shared_tasks')
        self.assertEqual(result['shared_task_ids'], [])
        self.assertEqual(result['a_only_task_ids'], ['a'])
        self.assertEqual(result['b_only_task_ids'], ['b'])
        self.assertEqual(result['both_passed'] + result['only_a_passed'] + result['only_b_passed'] + result['neither_passed'], 0)

    def test_tampered_tasks_rejected(self):
        path = self.root / (self.id + '.json')
        data = json.loads(path.read_text())
        data['tasks'][0]['expected'] = 'changed'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.plugin.report(self.id)

    def test_inspect_one_recorded_answer(self):
        self.plugin.record(self.id, 'a', 'model-a', 'yes', latency_ms=100)
        detail = self.plugin.result(self.id, 'a', 'model-a')
        self.assertEqual(detail['task']['prompt'], 'Say yes')
        self.assertEqual(detail['result']['answer'], 'yes')
        self.assertEqual(detail['result']['origin'], 'manual')
        with self.assertRaisesRegex(ValueError, 'No result'):
            self.plugin.result(self.id, 'b', 'model-a')

    def test_pending_recovery_requires_released_execution_lock(self):
        path = self.plugin.path(self.id)
        data = self.plugin.load(path)
        data['attempts'].append({'id': 'fixture', 'model': 'model-a', 'task_id': 'a', 'status': 'pending', 'seed': 0, 'lock_version': 1})
        self.plugin.save(path, data)
        with (self.root / (self.id + '.run.lock')).open('a') as active:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'still active'):
                self.plugin.recover_pending(self.id)
        self.assertEqual(self.plugin.recover_pending(self.id)['status'], 'unknown')
        self.assertEqual(self.plugin.report(self.id)['attempts'][0]['status'], 'unknown')
        self.plugin.abandon(self.id, 'a', 'model-a', 'Reviewed the server')
        missing = 'a' * 32
        with self.assertRaisesRegex(ValueError, 'Unknown bakeoff'):
            self.plugin.recover_pending(missing)
        self.assertFalse((self.root / (missing + '.run.lock')).exists())

if __name__ == '__main__':
    unittest.main()
