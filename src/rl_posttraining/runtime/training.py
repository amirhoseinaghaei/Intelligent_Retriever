"""Schedule snapshot groups using only learner, scorer, and environment interfaces."""
import random

from rl_posttraining.core.investigate import arguments_for
from .ports import Environments, Learner, Scorer
from .snapshot_data import load_records


def train_snapshots(args, prepared, dataset, learner: Learner, scorer: Scorer,
                    environments: Environments, log):
    rng = random.Random(args.seed)
    step, changed = 0, False
    for epoch in range(1, args.snapshot_epochs + 1):
        snapshots = dataset["snapshots"].copy()
        rng.shuffle(snapshots)
        for snapshot in snapshots:
            records = load_records(prepared, snapshot["id"])
            context = {"epoch": epoch, "snapshot_id": snapshot["id"], "snapshot": snapshot["name"],
                       "dump_sha256": snapshot["dump_sha256"], "schema_sha256": snapshot["schema_sha256"]}
            print(f"Loading snapshot {snapshot['name']} (epoch {epoch})", flush=True)
            with environments.open(snapshot["id"]) as endpoint:
                for _ in range(args.steps_per_snapshot):
                    record = rng.choice(records)
                    question = arguments_for(record)["text"]
                    queries = learner.sample(question)
                    log.group_started(context, step + 1, record, queries)
                    try:
                        results = scorer.score(queries, record, endpoint)
                    except Exception as error:
                        log.group_failed(context, step + 1, error)
                        raise
                    if len(results) != len(queries):
                        raise ValueError("Each sampled query requires one scoring result")
                    metrics = learner.update([result["reward"] for result in results])
                    step += 1
                    log.group(context, step, record, question, results, metrics)
                    print(f"group={step} snapshot={snapshot['id']} "
                          f"reward={metrics['mean_reward']:.3f} loss={metrics['loss']:.4f}", flush=True)
                # Persist completed work before moving to another environment.
                changed = learner.save(str(args.output / "adapter"))
            log.snapshot_finished(context, step)
    summary = {"backend": "ray-core-pytorch-grpo-artifactnet", "model": args.model,
               "steps": step, "snapshot_epochs": args.snapshot_epochs,
               "steps_per_snapshot": args.steps_per_snapshot,
               "snapshots": len(dataset["snapshots"]), "examples": dataset["examples"],
               "adapter_weights_changed": changed, "prepared": str(prepared),
               "reward": "live_evidence_relevance_to_original_investigate_request"}
    log.summary(summary)
    return summary
