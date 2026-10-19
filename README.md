# ArtifactNet snapshot RL training

Train one LoRA adapter with GRPO across archived problem/checkpoint graphs.
Each example is an investigate input (`x`) paired with its recorded retrieval
output (`y`). The model learns to rewrite the request text. Candidate rewrites
are executed against the matching Neo4j snapshot and scored by a relevance judge.

## Setup

Use Python 3.11+, Docker with Compose v2, and a working PyTorch/CUDA installation
for GPU training. `--cpu` explicitly enables slower CPU training.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config.example.toml config.toml
```

Fill in the judge API key and model in `config.toml`:

```toml
[reward]
provider = "openrouter"
api_key = "your-key"
model = "your-provider/model-id"
```

`OPENROUTER_API_KEY` and `OPENROUTER_JUDGE_MODEL` override file settings;
`--judge-model` overrides the model. The judge must support structured JSON
responses. Its credentials are passed separately from saved training arguments.
ArtifactNet also needs credentials for the embedding/LLM providers configured in
the archived project. The containers receive `OPENROUTER_API_KEY` and
`OPENAI_API_KEY` from the environment. Live retrieval and judging can incur API costs.

## Prepare, then train

Preparation requires no Docker daemon, model download, or API calls:

```bash
python train_llm.py --experiments ../AlphaSE-experiments --prepare-only \
  --output outputs/snapshot-data
```

Then train using the extracted files:

```bash
python train_llm.py --prepared outputs/snapshot-data/prepared \
  --snapshot-epochs 2 --steps-per-snapshot 5 --samples 4 \
  --output outputs/snapshot-grpo
```

To prepare and train in one command, pass `--experiments` instead of `--prepared`
and omit `--prepare-only`. Output directories must be new or empty. A source is
required: the repository contains only the snapshot experiment.

The training sequence is:

1. Extract investigate pairs and verify the graph dumps, schemas, and configurations.
2. Match examples by their exact source archive, including problem, run, and checkpoint.
3. Shuffle snapshot order each epoch and start one Neo4j/ArtifactNet environment.
4. Sample examples from that snapshot, generate candidate requests, score them, and update.
5. Save the adapter, remove the containers/network/database volume, and move to the next snapshot.

The same model and optimizer persist throughout the run. All candidates in a
GRPO group use the same snapshot. Examples are sampled with replacement; total
update groups equal `snapshots × snapshot-epochs × steps-per-snapshot`.

Only the request text is learned. `entryNodes`, `kTop`, `hops`, and `threshold`
remain as recorded. The judge scores actual retrieved node text against the
original request. Recorded `y` is retained for reference, excluded from prompts,
and is not an exact-match reward target. Empty retrievals earn 0; invalid text
earns -1. Known transient transport failures are retried up to three times, with
2- and 4-second delays. This includes the historical backend's HTTP 400
`fetch failed`, connection timeouts, and HTTP 429/502/503/504. The same candidate
and snapshot are used on each attempt. Retries can repeat paid upstream calls.
Invalid internal LLM output after ArtifactNet's repair attempt is also retried
within the same three-attempt limit. This includes omitted schema tag results;
the trainer never fabricates missing tags or penalizes a candidate for this error.
Request/schema validation errors, or provider failures after the retry limit, stop
training and trigger environment cleanup; they never become negative rewards.
The graph is the snapshot at capture time; earlier calls may have seen less data.

## Prepared data and outputs

```text
outputs/snapshot-data/prepared/
  dataset.json                       Snapshot IDs, counts, and skipped examples
  pairs/<problem>/<checkpoint>/      Original extracted training_data.json files
  environments/<snapshot-id>/
    backup/neo4j.dump                 Verified immutable snapshot
    project/.artnet/config.yaml       Archived provider settings; editable
    project/.artnet/schema.yaml       Exact archived schema
    training_data.json               Examples belonging to this archive only
    compose.json                     Isolated Neo4j and ArtifactNet services
    stack.json                       Image, volume, and checksum metadata
    .env                             Generated local Neo4j password
```

The prepared folder contains private archived configuration: keep it out of Git.
Keep it in its original location because Compose mounts use absolute paths.
Existing prepared data remains compatible with the refactored trainer.

Edit each extracted `project/.artnet/config.yaml` if provider settings need to
change. For a custom credential variable, also add its environment mapping in
`compose.json`. Dumps and schemas are checksum-verified on reuse. Only snapshots
with valid matching training examples are prepared; failures and skipped records
are listed in the extraction summary and `dataset.json`.

Images default to references in the archive manifest. If unavailable or built
for a different CPU architecture, supply compatible images during preparation:

```bash
python train_llm.py --experiments ../AlphaSE-experiments --prepare-only \
  --neo4j-image YOUR_COMPATIBLE_NEO4J_IMAGE \
  --artifactnet-backend-image YOUR_COMPATIBLE_ARTIFACTNET_IMAGE \
  --output outputs/snapshot-data-new
```

The existing 90 prepared environments now use the local image
`artnet-backend:snapshot-c0e442f` with `pull_policy: never`. It was built from
ArtifactNet revision `c0e442f` (package 1.4.3), whose bundled schema exactly
matches the failing checkpoint. Both schema variants in the prepared dataset
validate with it, and the failing checkpoint passed live readiness and exact-node
retrieval checks. This is a compatible historical build, not the original image
identified by the archive manifest. The original manifest and schemas are retained.
The local `artnet-backend:latest` (package 1.4.4) rejects the archived node-level
`relation_inference` fields and cannot run these checkpoints unchanged.

To rebuild the compatible image from the sibling ArtifactNet repository:

```bash
git -C ../ArtifactNet archive c0e442f | docker build \
  --target artifactnet-backend -t artnet-backend:snapshot-c0e442f -
```

`--port` (default 8001) and `--bolt-port` (default 7687) are also preparation-time
settings. `--snapshot-timeout` controls startup readiness; `--artifactnet-timeout`
controls each investigate attempt. The snapshot schema and ArtifactNet storage
format must match the selected image. Health checks verify schema loading and
connectivity; an actual retrieval verifies graph compatibility.

Training writes:

- `adapter/`: latest completed snapshot's adapter and tokenizer.
- `rollouts.jsonl`: queries, retrievals, rewards, advantages, and source provenance.
- `requests.jsonl`: candidate groups saved before calling ArtifactNet, including failed groups.
- `failures.jsonl`: scoring failures with snapshot, step, and backend error details.
- `metrics.jsonl`: GRPO metrics per group, with snapshot IDs and checksums.
- `snapshots.jsonl`: completed snapshots and successful environment cleanup.
- `config.json` and `summary.json`: run settings and final counts.

There is currently no held-out evaluation, best-adapter selection, or optimizer
resume. Equal rewards within a group give zero retrieval advantages; inspect
reward variance before assuming more steps will help. Default model:
`Qwen/Qwen2.5-7B-Instruct`; default generation length: 128 tokens.

## Inspect snapshots manually

The manager can run independently of training:

```bash
python scripts/checkpoint_manager.py index ../AlphaSE-experiments
python scripts/checkpoint_manager.py start CHECKPOINT_ID
python scripts/checkpoint_manager.py status CHECKPOINT_ID
python scripts/checkpoint_manager.py stop CHECKPOINT_ID
python scripts/checkpoint_manager.py stop CHECKPOINT_ID --remove-data
```

Standalone state defaults to `.checkpoint_manager/`. To inspect the trainer's
prepared environments, pass `--state-dir outputs/snapshot-data/prepared/environments`
**before** the subcommand. Run manager commands sequentially. `stop` retains the
restored database unless `--remove-data` is supplied; training always removes it.
After an interrupted restore, use `stop CHECKPOINT_ID --remove-data` before retrying.
If Docker fails during automatic cleanup, the same command removes the leftover stack.
When service startup fails, `startup-error.log` is saved inside the prepared
environment before cleanup, including the health endpoint's error explanation.

ArtifactNet is exposed on localhost at `http://127.0.0.1:8001/artifactnet` and
Neo4j at `bolt://127.0.0.1:7687`. The graph database is restored offline into a
fresh volume; `system.dump` is not restored, so original users are not imported.
Use retrieval operations when inspecting: the backend also exposes mutation tools.

```bash
curl --fail-with-body http://127.0.0.1:8001/artifactnet/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"tool":"artifactnet_investigate","arguments":{"text":"Find validation requirements","entryNodes":"Requirement","kTop":5,"hops":1,"threshold":0.5}}'
```

Inspect startup failures with:

```bash
docker compose -f .checkpoint_manager/CHECKPOINT_ID/compose.json logs --tail 100
```

To extract only recorded pairs without preparing graph snapshots:

```bash
python scripts/extract_investigate_data.py ../AlphaSE-experiments \
  --output training_data/new-export
```

## Code responsibilities

The code uses small protocols at the training boundary, with concrete adapters
for Ray and Docker. No dependency-injection framework or class hierarchy is required.

| Location | Job |
| --- | --- |
| `cli/` | Parse commands and report user-facing errors. |
| `config.py` | Load and validate judge credentials. |
| `core/investigate.py` | Build prompts and validate investigate arguments. |
| `core/grpo/` | Sample completions, calculate losses, and update LoRA weights. |
| `runtime/snapshot_data.py` | Assemble and validate examples matched to snapshots. |
| `runtime/experiment.py` | Compose preparation, credentials, and training resources. |
| `runtime/training.py` | Schedule snapshot groups through learner/scorer/environment protocols. |
| `runtime/ports.py` | Declare the three small interfaces consumed by the loop. |
| `runtime/artifacts.py` | Write training records and summaries. |
| `integrations/experiment_data.py` | Read investigate calls from agent sessions. |
| `integrations/snapshots.py` | Index archives and verify/extract snapshot files. |
| `integrations/compose.py` | Build Compose service definitions. |
| `integrations/checkpoints.py` | Restore databases and start/stop Docker services. |
| `integrations/environments.py` | Guarantee checkpoint environment cleanup. |
| `integrations/artifactnet.py` | Execute investigate requests and collect graph evidence. |
| `integrations/reward_calculator.py` | Request and validate relevance judgments. |
| `integrations/models.py` | Load the tokenizer, model, and LoRA adapter. |
| `integrations/ray_training.py` | Adapt synchronous training calls to Ray actors/tasks. |

Core optimization knows nothing about ArtifactNet or Docker. The scheduling loop
knows nothing about Ray. Model loading and external services are imported lazily,
so preparation and `--help` work without training dependencies.

## Verification

```bash
make test
make check
```

Tests cover archive verification, exact snapshot routing, cleanup on failure,
credential isolation, API handling, Ray adapters, GRPO math, and a real tiny
Qwen/PEFT backward pass and adapter reload. They need the training dependencies
but make no API calls, download no models, and start no Docker/Ray services.
