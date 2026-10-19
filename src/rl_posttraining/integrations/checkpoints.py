"""Restore, start, and remove isolated Neo4j/ArtifactNet containers."""
import hashlib
import json
import subprocess

from .files import write_json
from .diagnostics import save_startup_diagnostics
from .snapshots import prepare, saved_stack


def check_docker():
    """Fail before allocating a model if Docker or Compose is unavailable."""
    commands = [('docker', 'version', '--format', '{{.Server.Version}}'),
                ('docker', 'compose', 'version')]
    for command in commands:
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError('Docker preflight failed. Check that docker ps and docker compose version work.') from error
        if result.returncode:
            if 'permission denied' in result.stderr.lower():
                raise RuntimeError('Docker socket permission denied. Run docker ps as your normal user; '
                                   'ask the server administrator to grant Docker access and start a new login session.')
            detail = result.stderr.strip()[:1000]
            raise RuntimeError(f'Docker preflight failed: {detail}')


def run(*args, capture=False):
    return subprocess.run(args, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def compose(folder, *args, capture=False):
    return run('docker', 'compose', '-f', str(folder / 'compose.json'), *args, capture=capture)


def start(state, identity, args):
    folder, metadata = prepare(state, identity, args)
    # Keep resource use predictable: one checkpoint per manager directory.
    projects = run('docker', 'ps', '--filter', 'label=rl.checkpoint.owner=' + metadata['owner'],
                   '--format', '{{.Label "com.docker.compose.project"}}', capture=True).splitlines()
    if any(project != metadata['project'] for project in projects):
        raise ValueError('Another checkpoint is running. Stop it before starting this one.')
    if not metadata['restored']:
        existing = run('docker', 'volume', 'ls', '--format', '{{.Name}}', capture=True).splitlines()
        if metadata['volume'] in existing:
            raise ValueError('An unfinished restore volume exists. Run stop ID --remove-data, then retry.')
        run('docker', 'volume', 'create', metadata['volume'])
        # Never restore system.dump: a fresh system database gives this stack its
        # own credentials. The original checkpoint's users are not needed.
        run('docker', 'run', '--rm', '--network', 'none',
            '--mount', f'type=volume,src={metadata["volume"]},dst=/data',
            '--mount', f'type=bind,src={folder / "backup"},dst=/backup,readonly',
            '--entrypoint', 'neo4j-admin', metadata['neo4j_image'],
            'database', 'load', 'neo4j', '--from-path=/backup')
        metadata['restored'] = True
        write_json(folder / 'stack.json', metadata)
    try:
        compose(folder, 'up', '-d', '--wait', '--wait-timeout', str(args.timeout))
    except subprocess.CalledProcessError as error:
        try:
            diagnostics = save_startup_diagnostics(folder, metadata['backend_port'])
        except Exception as diagnostics_error:
            error.add_note(f'Could not save startup diagnostics: {diagnostics_error}')
            raise
        raise RuntimeError(f'Checkpoint services failed to become healthy. See {diagnostics}') from error
    print(f'Checkpoint: {metadata["name"]}')
    print(f'ArtifactNet: http://127.0.0.1:{metadata["backend_port"]}/artifactnet')
    print(f'Neo4j: bolt://127.0.0.1:{metadata["bolt_port"]}')
    print(f'Local Neo4j password: {folder / ".env"}')
    return metadata


def stop(state, identity, remove_data=False):
    """Remove the containers/network; optionally discard the restored volume."""
    folder, metadata = saved_stack(state, identity)
    compose(folder, 'down')
    if remove_data:
        existing = run('docker', 'volume', 'ls', '--format', '{{.Name}}', capture=True).splitlines()
        if metadata['volume'] in existing:
            run('docker', 'volume', 'rm', metadata['volume'])
        metadata['restored'] = False
        write_json(folder / 'stack.json', metadata)


def status(state, identity=None):
    """Show saved checkpoint identities and actual Docker container state."""
    if identity:
        folder, _ = saved_stack(state, identity)
        compose(folder, 'ps', '--all')
        return
    for checkpoint_id, entry in json.loads((state / 'catalog.json').read_text()).items():
        print(checkpoint_id, entry['name'])
    owner = hashlib.sha256(str(state).encode()).hexdigest()[:12]
    run('docker', 'ps', '-a', '--filter', 'label=rl.checkpoint.owner=' + owner)


