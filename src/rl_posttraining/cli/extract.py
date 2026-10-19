"""Extract investigate input/output pairs from experiment archives."""
import argparse
import json
from pathlib import Path

from rl_posttraining.integrations.experiment_data import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', help='Experiments folder to search recursively')
    parser.add_argument('--source', type=Path, help='Alternative to the positional folder argument')
    parser.add_argument('--output', type=Path,
                        help='New output folder (default: training_data/<source-folder-name>)')
    args = parser.parse_args()
    if (args.folder is None) == (args.source is None):
        parser.error('Provide one experiments folder, either positionally or with --source')
    source = (args.folder or args.source).expanduser().resolve()
    destination = (args.output or Path('training_data') / source.name).expanduser().resolve()
    try:
        summary = extract(source, destination)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Output: {destination}')
    print(json.dumps({k: v for k, v in summary.items() if k != 'archives_detail'}, indent=2))
    if summary['errors']:
        parser.exit(1, 'Extraction finished with errors; see summary.json for details.\n')


if __name__ == "__main__":
    main()
