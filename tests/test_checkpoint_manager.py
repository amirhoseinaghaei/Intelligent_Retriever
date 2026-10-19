"""Offline checks for snapshot isolation and restore safety; Docker is mocked."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile


from rl_posttraining.integrations import checkpoints as manager, snapshots


class CheckpointManagerTests(unittest.TestCase):
    def test_preflight_reports_permission_error_without_starting_containers(self):
        denied = subprocess.CompletedProcess(['docker'], 1, stdout='', stderr='permission denied: docker.sock')
        with patch.object(manager.subprocess, 'run', return_value=denied) as run:
            with self.assertRaisesRegex(RuntimeError, 'Docker socket permission denied'):
                manager.check_docker()
        run.assert_called_once()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / 'state'
        self.state.mkdir()
        self.archive = self.root / 'experiments/problem/run-a/checkpoint_1/experiment.zip'
        self.archive.parent.mkdir(parents=True)
        self.args = argparse.Namespace(neo4j_image=None, artifactnet_backend_image=None,
                                       bolt_port=7687, port=8001, timeout=240)
        self.make_archive()
        with patch('builtins.print'):
            snapshots.index(self.root / 'experiments', self.state)
        self.identity = next(iter(json.loads((self.state / 'catalog.json').read_text())))

    def make_archive(self, corrupt=False):
        files = {'services/neo4j.dump': b'graph',
                 snapshots.ARTNET + 'schema.yaml': b'version: test',
                 snapshots.ARTNET + 'config.yaml': b'schema: .artnet/schema.yaml'}
        manifest = {'files': {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
                    'images': {'neo4j': {'digests': ['neo4j@sha256:example']},
                               'artifactnet-backend': {'id': 'sha256:backend'}}}
        with ZipFile(self.archive, 'w') as zipped:
            zipped.writestr('manifest.json', json.dumps(manifest))
            for name, data in files.items():
                zipped.writestr(name, b'corrupt' if corrupt and name.endswith('.dump') else data)
            zipped.writestr('../../outside', 'must never be extracted')
            zipped.writestr('services/system.dump', 'old credentials')

    def test_prepare_verifies_and_extracts_only_required_members(self):
        folder, metadata = manager.prepare(self.state, self.identity, self.args)
        self.assertFalse((self.root / 'outside').exists())
        self.assertEqual(list((folder / 'backup').iterdir()), [folder / 'backup/neo4j.dump'])
        self.assertEqual(metadata['neo4j_image'], 'neo4j@sha256:example')
        compose = json.loads((folder / 'compose.json').read_text())
        self.assertTrue(compose['volumes']['data']['external'])
        self.assertIn(':ro', compose['services']['artifactnet']['volumes'][0])
        self.assertEqual((folder / '.env').stat().st_mode & 0o777, 0o600)

    def test_corrupt_dump_never_becomes_prepared_stack(self):
        self.make_archive(corrupt=True)
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            manager.prepare(self.state, self.identity, self.args)
        self.assertFalse((self.state / self.identity / 'stack.json').exists())

    def test_failed_restore_is_not_marked_successful(self):
        def fake_run(*args, **kwargs):
            if args[:2] == ('docker', 'run'):
                raise subprocess.CalledProcessError(1, args)
            return ''
        with patch.object(manager, 'run', side_effect=fake_run):
            with self.assertRaises(subprocess.CalledProcessError):
                manager.start(self.state, self.identity, self.args)
        _, metadata = manager.saved_stack(self.state, self.identity)
        self.assertFalse(metadata['restored'])

    def test_existing_volume_is_never_overwritten(self):
        _, metadata = manager.prepare(self.state, self.identity, self.args)
        with patch.object(manager, 'run', side_effect=['', metadata['volume'] + '\n']) as run:
            with self.assertRaisesRegex(ValueError, 'unfinished restore'):
                manager.start(self.state, self.identity, self.args)
            self.assertEqual(run.call_count, 2)

    def test_running_other_checkpoint_blocks_start(self):
        with patch.object(manager, 'run', return_value='other-project\n'):
            with self.assertRaisesRegex(ValueError, 'Another checkpoint'):
                manager.start(self.state, self.identity, self.args)

    def test_resume_does_not_reload_dump(self):
        folder, metadata = manager.prepare(self.state, self.identity, self.args)
        metadata['restored'] = True
        manager.write_json(folder / 'stack.json', metadata)
        with patch.object(manager, 'run', return_value='') as run, patch('builtins.print'):
            manager.start(self.state, self.identity, self.args)
        self.assertEqual(run.call_count, 2)
        self.assertIn('up', run.call_args.args)

    def test_unhealthy_start_preserves_diagnostics_before_cleanup(self):
        folder, metadata = manager.prepare(self.state, self.identity, self.args)
        metadata['restored'] = True
        manager.write_json(folder / 'stack.json', metadata)
        error = subprocess.CalledProcessError(1, ['docker', 'compose', 'up'])
        path = folder / 'startup-error.log'
        with patch.object(manager, 'run', return_value=''), \
                patch.object(manager, 'compose', side_effect=error), \
                patch.object(manager, 'save_startup_diagnostics', return_value=path) as save:
            with self.assertRaisesRegex(RuntimeError, 'startup-error.log'):
                manager.start(self.state, self.identity, self.args)
        save.assert_called_once_with(folder, metadata['backend_port'])


if __name__ == '__main__':
    unittest.main()
