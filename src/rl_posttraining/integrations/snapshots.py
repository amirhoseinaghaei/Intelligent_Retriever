"""Catalog and verify immutable checkpoint files; never starts Docker."""
import hashlib
import json
import re
import secrets
from zipfile import ZipFile

from .compose import build_compose
from .files import write_json


ARTNET = 'checkout/.agent_workspace/artifactnet/.artnet/'
MEMBERS = {'services/neo4j.dump': 'backup/neo4j.dump',
           ARTNET + 'schema.yaml': 'project/.artnet/schema.yaml',
           ARTNET + 'config.yaml': 'project/.artnet/config.yaml'}


def index(source, state):
    """Catalog only archives that actually contain a Neo4j snapshot."""
    if not source.is_dir():
        raise ValueError(f'Experiments folder does not exist: {source}')
    entries = {}
    for archive in sorted(source.rglob('*.zip')):
        if archive.name not in {'experiment.zip', 'experiments.zip'}:
            continue
        with ZipFile(archive) as zipped:
            if 'services/neo4j.dump' not in zipped.namelist():
                continue
        # Full relative paths distinguish configurations, runs, and checkpoints.
        relative = archive.relative_to(source).as_posix()
        identity = hashlib.sha256(str(archive).encode()).hexdigest()[:12]
        entries[identity] = {'archive': str(archive), 'name': relative}
        print(f'{identity}  {relative}')
    if not entries:
        raise ValueError('No experiment archives containing services/neo4j.dump were found')
    write_json(state / 'catalog.json', entries)
    print(f'Indexed {len(entries)} snapshots.')
    return entries


def saved_stack(state, identity):
    if not re.fullmatch(r'[a-f0-9]{12}', identity):
        raise ValueError('Use the 12-character checkpoint ID printed by index')
    folder = state / identity
    metadata = folder / 'stack.json'
    if not metadata.exists():
        raise ValueError(f'Checkpoint {identity} has not been prepared; run start first')
    return folder, json.loads(metadata.read_text())


def image_reference(manifest, service, override):
    if override:
        return override
    image = manifest.get('images', {}).get(service, {})
    references = image.get('digests', [])
    reference = references[0] if references else image.get('id') or image.get('tag')
    if not reference:
        raise ValueError(f'Archive has no {service} image; supply --{service}-image')
    return reference


def prepare(state, identity, args):
    """Extract a small allowlist, checking every extracted byte against the manifest."""
    catalog = json.loads((state / 'catalog.json').read_text())
    if identity not in catalog:
        raise ValueError(f'Unknown checkpoint ID: {identity}')
    folder = state / identity
    if (folder / 'stack.json').exists():
        return saved_stack(state, identity)
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)  # Original configuration can contain provider credentials.
    with ZipFile(catalog[identity]['archive']) as zipped:
        manifest = json.loads(zipped.read('manifest.json'))
        for member, target in MEMBERS.items():
            expected = manifest.get('files', {}).get(member)
            if not expected:
                raise ValueError(f'Missing manifest checksum for {member}')
            destination = folder / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            # Stream the dump instead of holding the database in Python memory.
            with zipped.open(member) as source, destination.open('wb') as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                destination.unlink()
                raise ValueError(f'Checksum mismatch: {member}')
    password = secrets.token_hex(24)
    owner = hashlib.sha256(str(state).encode()).hexdigest()[:12]
    project = 'rl-checkpoint-' + owner + '-' + identity
    neo4j_image = image_reference(manifest, 'neo4j', args.neo4j_image)
    backend_image = image_reference(manifest, 'artifactnet-backend', args.artifactnet_backend_image)
    metadata = dict(catalog[identity], project=project, owner=owner,
                    volume=project + '-data', restored=False,
                    bolt_port=args.bolt_port, backend_port=args.port,
                    neo4j_image=neo4j_image, backend_image=backend_image,
                    schema_sha256=manifest['files'][ARTNET + 'schema.yaml'],
                    dump_sha256=manifest['files']['services/neo4j.dump'])
    # Compose reads runtime secrets from a private file, not from command arguments.
    (folder / '.env').write_text('CHECKPOINT_PASSWORD=' + password + '\n')
    (folder / '.env').chmod(0o600)
    configuration = build_compose(folder, metadata)
    write_json(folder / 'compose.json', configuration)
    write_json(folder / 'manifest.json', manifest)
    write_json(folder / 'stack.json', metadata)
    return folder, metadata


