"""Build the two-service Compose definition for one isolated snapshot."""


def build_compose(folder, metadata):
    environment = {
        'NEO4J_PASSWORD': '${CHECKPOINT_PASSWORD}',
        'ARTNET_BOOTSTRAP_PROJECT': '0',
        'ARTIFACTNET_BACKEND_PROJECT_ROOT': '/workspace',
        'ARTIFACTNET_BACKEND_HOST': '0.0.0.0',
        'ARTIFACTNET_BACKEND_PORT': '8001',
        'ARTIFACTNET_BACKEND_HEALTH_PATH': '/health',
        'ARTIFACTNET_BACKEND_REDIS_URL': '',
        'ARTIFACTNET_BACKEND_NEO4J_URI': 'bolt://neo4j:7687',
        'ARTIFACTNET_BACKEND_NEO4J_USER': 'neo4j',
        'ARTIFACTNET_BACKEND_NEO4J_PASSWORD': '${CHECKPOINT_PASSWORD}',
        'ARTIFACTNET_BACKEND_NEO4J_DATABASE': 'neo4j',
    }
    # Forward credentials only when supplied by the caller; don't save their values.
    for key in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY'):
        environment[key] = '${' + key + ':-}'
    configuration = {
        'name': metadata['project'],
        'services': {
            'neo4j': {
                'image': metadata['neo4j_image'],
                'labels': {'rl.checkpoint.owner': metadata['owner']},
                'environment': {'NEO4J_AUTH': 'neo4j/${CHECKPOINT_PASSWORD}'},
                'ports': [f"127.0.0.1:{metadata['bolt_port']}:7687"],
                'volumes': ['data:/data'],
                'healthcheck': {
                    'test': ['CMD-SHELL', 'cypher-shell -u neo4j -p "$${NEO4J_AUTH#*/}" "RETURN 1" >/dev/null 2>&1'],
                    'interval': '5s', 'timeout': '10s', 'retries': 30,
                },
            },
            'artifactnet': {
                'image': metadata['backend_image'],
                'labels': {'rl.checkpoint.owner': metadata['owner']},
                'command': ['artnet-backend'],
                'depends_on': {'neo4j': {'condition': 'service_healthy'}},
                'environment': environment,
                'volumes': [str(folder / 'project') + ':/workspace:ro'],
                'ports': [f"127.0.0.1:{metadata['backend_port']}:8001"],
                'healthcheck': {
                    'test': ['CMD', 'node', '-e', "fetch('http://127.0.0.1:8001/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"],
                    'interval': '5s', 'timeout': '10s', 'retries': 30,
                },
            },
        },
        'volumes': {'data': {'external': True, 'name': metadata['volume']}},
    }
    return configuration
