#!/usr/bin/env python3
"""Extract investigate input/output pairs without unpacking experiment archives."""
from collections import defaultdict
import json
from zipfile import BadZipFile, ZipFile


def extract(source, destination):
    if not source.is_dir():
        raise ValueError(f'Source folder does not exist: {source}')
    if destination.exists():
        raise ValueError(f'Output already exists; choose a new --output folder: {destination}')
    groups = defaultdict(list)
    reports = []
    archives = sorted(p for p in source.rglob('*.zip')
                      if p.name in {'experiment.zip', 'experiments.zip'})
    if not archives:
        raise ValueError(f'No experiment.zip or experiments.zip archives found under {source}')
    for archive in archives:
        relative = archive.relative_to(source)
        parts = relative.parts
        checkpoint = next((p for p in parts if p.startswith('checkpoint_')), 'uncheckpointed')
        run_index = next((i for i, p in enumerate(parts) if p.startswith('run-')), None)
        checkpoint_index = next((i for i, p in enumerate(parts) if p.startswith('checkpoint_')), None)
        problem = (parts[run_index - 1] if run_index is not None and run_index > 0
                   else parts[checkpoint_index - 1] if checkpoint_index is not None and checkpoint_index > 0
                   else archive.parent.name)
        records = groups[problem, checkpoint]
        report = dict(archive=str(relative), problem=problem, checkpoint=checkpoint,
                      sessions=0, pairs=0, duplicates=0, excluded_calls=[], errors=[])
        seen = {}
        try:
            with ZipFile(archive) as zipped:
                names = sorted(n for n in zipped.namelist()
                               if n.startswith('checkout/.agent_workspace/agents/')
                               and '/sessions/' in n and n.endswith('.json'))
                for name in names:
                    try:
                        session = json.loads(zipped.read(name))
                        report['sessions'] += 1
                        for message in session.get('messages', []):
                            for part in message.get('parts', []):
                                tool = part.get('tool', '')
                                if part.get('type') != 'tool' or tool.lower().split('_')[-1] != 'investigate':
                                    continue
                                state = part.get('state', {})
                                call_id = part.get('callID') or part.get('id')
                                identity = (session.get('session_id'), call_id)
                                if state.get('status') != 'completed' or 'input' not in state or 'output' not in state:
                                    report['excluded_calls'].append(dict(session_file=name, call_id=call_id,
                                                                        status=state.get('status')))
                                    continue
                                pair = dict(x=state['input'], y=state['output'])
                                if call_id and identity in seen:
                                    if seen[identity] != pair:
                                        raise ValueError(f'Conflicting snapshots for {identity}')
                                    report['duplicates'] += 1
                                    continue
                                seen[identity] = pair
                                records.append(dict(**pair, metadata=dict(
                                    archive=str(relative), session_file=name,
                                    run=parts[run_index] if run_index is not None else None,
                                    configuration='/'.join(parts[:run_index-1]) if run_index is not None else '/'.join(parts[:-2]),
                                    agent=name.split('/')[3], role=session.get('role'),
                                    session_id=session.get('session_id'), call_id=call_id,
                                    message_id=message.get('info', {}).get('id'), tool=tool,
                                    truncated=state.get('metadata', {}).get('truncated', False))))
                                report['pairs'] += 1
                    except (ValueError, KeyError, TypeError) as error:
                        report['errors'].append(dict(session_file=name, error=str(error)))
        except (OSError, ValueError, BadZipFile) as error:
            report['errors'].append(dict(error=str(error)))
        reports.append(report)
    destination.mkdir(parents=True, exist_ok=False)
    for (problem, checkpoint), records in sorted(groups.items()):
        folder = destination / problem / checkpoint
        folder.mkdir(parents=True)
        (folder / 'training_data.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
    summary = dict(source=str(source), archives=len(archives), problems=len({p for p, _ in groups}),
                   checkpoint_files=len(groups), pairs=sum(len(r) for r in groups.values()),
                   sessions=sum(r['sessions'] for r in reports),
                   excluded_calls=sum(len(r['excluded_calls']) for r in reports),
                   errors=sum(len(r['errors']) for r in reports), archives_detail=reports)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (destination / 'README.txt').write_text(
        'Each problem/checkpoint/training_data.json is a JSON array of paired calls.\n'
        'x: exact investigate call input object. y: exact retrieval output (usually a JSON string).\n'
        'Parse y with json.loads when structured retrieval results are needed.\n'
        'Metadata identifies the source archive, run, agent, session, and call.\n'
        'All configurations/runs are retained; repeated call snapshots within an archive are deduplicated.\n'
        'Calls repeated across checkpoints remain present to preserve checkpoint provenance.\n'
        'Completed empty retrievals and truncated outputs are retained, with truncation flagged.\n'
        'Failed/incomplete calls are excluded and listed in summary.json; empty checkpoint files are retained.\n'
        'Archives without checkpoint directories are grouped under uncheckpointed.\n')
    return summary


