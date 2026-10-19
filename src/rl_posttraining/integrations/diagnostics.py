"""Preserve startup errors before failed checkpoint containers are removed."""
import os
import subprocess
import urllib.error
import urllib.request


def save_startup_diagnostics(folder, port):
    sections = []
    command = ['docker', 'compose', '-f', str(folder / 'compose.json'),
               'logs', '--no-color', '--tail', '100']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        sections.append(result.stdout + result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        sections.append(f'Could not collect service logs: {error}')
    # Readiness errors are returned as HTTP 503 bodies, not always service logs.
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
            sections.append('ArtifactNet health: ' + response.read().decode())
    except urllib.error.HTTPError as error:
        sections.append(f'ArtifactNet health HTTP {error.code}: ' + error.read().decode())
    except (OSError, urllib.error.URLError) as error:
        sections.append(f'ArtifactNet health unavailable: {error}')
    secrets = [value for name, value in os.environ.items()
               if any(word in name.upper() for word in ('KEY', 'TOKEN', 'PASSWORD')) and len(value) > 3]
    for line in (folder / '.env').read_text().splitlines():
        if '=' in line:
            secrets.append(line.split('=', 1)[1])
    text = '\n\n'.join(sections)
    for secret in secrets:
        if secret:
            text = text.replace(secret, '[REDACTED]')
    path = folder / 'startup-error.log'
    path.write_text(text + '\n')
    path.chmod(0o600)
    return path
