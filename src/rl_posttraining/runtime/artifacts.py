"""Write training records without making training decisions."""
from contextlib import ExitStack
import json

from rl_posttraining.integrations.files import write_json


class TrainingLog:
    def __init__(self, output):
        self.output = output
        self.files = ExitStack()

    def __enter__(self):
        try:
            self.rollouts = self.files.enter_context((self.output / "rollouts.jsonl").open("w"))
            self.metrics = self.files.enter_context((self.output / "metrics.jsonl").open("w"))
            self.snapshots = self.files.enter_context((self.output / "snapshots.jsonl").open("w"))
            self.requests = self.files.enter_context((self.output / "requests.jsonl").open("w"))
            self.failures = self.files.enter_context((self.output / "failures.jsonl").open("w"))
        except BaseException:
            self.files.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.files.__exit__(*exc)

    @staticmethod
    def append(stream, value):
        stream.write(json.dumps(value) + "\n")
        stream.flush()

    def group(self, context, step, record, question, results, metrics):
        if len(results) != len(metrics["advantages"]):
            raise ValueError("Each scored query requires one advantage")
        for result, advantage in zip(results, metrics["advantages"]):
            self.append(self.rollouts, {**context, "step": step, "question": question,
                                       "source": record["metadata"], "original_input": record["x"],
                                       "advantage": advantage, **result})
        self.append(self.metrics, {**context, "step": step, **metrics})

    def group_started(self, context, step, record, queries):
        # Persist candidates before remote calls so a failed group is reproducible.
        self.append(self.requests, {**context, "step": step, "source": record["metadata"],
                                    "original_input": record["x"], "queries": queries})

    def group_failed(self, context, step, error):
        self.append(self.failures, {**context, "step": step, "error": str(error)})

    def snapshot_finished(self, context, step):
        self.append(self.snapshots, {**context, "last_step": step, "environment_removed": True})

    def summary(self, summary):
        write_json(self.output / "summary.json", summary)
