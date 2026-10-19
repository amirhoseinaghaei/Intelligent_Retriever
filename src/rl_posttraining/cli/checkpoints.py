"""Command-line access to snapshot indexing and Docker lifecycle operations."""
import argparse
from pathlib import Path
import subprocess
from zipfile import BadZipFile

from rl_posttraining.integrations.checkpoints import start, status, stop
from rl_posttraining.integrations.snapshots import index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, default=Path('.checkpoint_manager'))
    commands = parser.add_subparsers(dest='command', required=True)
    indexing = commands.add_parser('index', help='Discover checkpoint archives; no Docker needed')
    indexing.add_argument('folder', type=Path)
    starting = commands.add_parser('start', help='Restore and start one checkpoint, or resume it')
    starting.add_argument('id')
    starting.add_argument('--neo4j-image', help='Override archived Neo4j image on first start')
    starting.add_argument('--artifactnet-backend-image', help='Override archived ArtifactNet image on first start')
    starting.add_argument('--port', type=int, default=8001)
    starting.add_argument('--bolt-port', type=int, default=7687)
    starting.add_argument('--timeout', type=int, default=240)
    status_command = commands.add_parser('status', help='Show indexed IDs and actual container state')
    status_command.add_argument('id', nargs='?')
    stopping = commands.add_parser('stop', help='Stop a checkpoint, retaining its restored database')
    stopping.add_argument('id')
    stopping.add_argument('--remove-data', action='store_true', help='Also delete this checkpoint volume for a fresh restore')
    args = parser.parse_args()
    state = args.state_dir.expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    state.chmod(0o700)
    try:
        if args.command == 'index':
            index(args.folder.expanduser().resolve(), state)
        elif args.command == 'start':
            if not (1 <= args.port <= 65535 and 1 <= args.bolt_port <= 65535) or args.port == args.bolt_port:
                raise ValueError('Use distinct ports between 1 and 65535')
            start(state, args.id, args)
        elif args.command == 'status':
            status(state, args.id)
        elif args.command == 'stop':
            stop(state, args.id, remove_data=args.remove_data)
    except (OSError, ValueError, KeyError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
