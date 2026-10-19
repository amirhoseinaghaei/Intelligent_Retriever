"""Keep the readiness explanation, but never persist known runtime secrets."""
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from rl_posttraining.integrations.diagnostics import save_startup_diagnostics


class StartupDiagnosticsTests(unittest.TestCase):
    def test_health_error_is_saved_and_credentials_are_redacted(self):
        logs = subprocess.CompletedProcess([], 0, 'backend password local-password', '')
        error = urllib.error.HTTPError('http://localhost', 503, 'unhealthy', {},
                                      io.BytesIO(b'Invalid KG schema; api-secret'))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / '.env').write_text('CHECKPOINT_PASSWORD=local-password\n')
            with patch('subprocess.run', return_value=logs), \
                    patch('urllib.request.urlopen', side_effect=error), \
                    patch.dict(os.environ, {'OPENROUTER_API_KEY': 'api-secret'}):
                path = save_startup_diagnostics(folder, 8001)
            contents = path.read_text()
            self.assertIn('HTTP 503', contents)
            self.assertIn('Invalid KG schema', contents)
            self.assertNotIn('local-password', contents)
            self.assertNotIn('api-secret', contents)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
