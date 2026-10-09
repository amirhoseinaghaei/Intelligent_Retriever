"""Tiny, deterministic retrieval task; usable without training dependencies."""
import re

DOCUMENTS = {
    "access": "Access keys expire after 90 days. Rotate credentials in the security console.",
    "backup": "Snapshots are retained for 30 days. Restore a backup from the storage console.",
    "billing": "Invoices are issued monthly. Download a receipt from the billing console.",
    "limits": "API requests are limited to 100 per minute. Retry with exponential backoff.",
    "deploy": "Releases use rolling deployment. Roll back a release from the deployment console.",
    "logs": "Application logs are retained for 7 days. Search logs in the observability console.",
}
# Labels belong only to reward/evaluation code, never to observations.
TRAIN = [
    ("When do access keys expire?", "access"),
    ("How long are snapshots retained?", "backup"),
    ("Where do I download an invoice?", "billing"),
    ("How many API requests per minute?", "limits"),
    ("How do I roll back a release?", "deploy"),
    ("Where can I search application logs?", "logs"),
    ("How do I renew a password?", "access"),
    ("Can I recover an archive?", "backup"),
    ("Where is my payment statement?", "billing"),
    ("What is the traffic quota?", "limits"),
    ("How do I undo a rollout?", "deploy"),
    ("Where can I inspect telemetry?", "logs"),
]
TEST = [
    ("When should I renew my password?", "access"),
    ("How long can I recover an archive?", "backup"),
    ("How can I download my payment statement?", "billing"),
    ("What happens after exceeding the traffic quota?", "limits"),
    ("Can I undo yesterday's rollout?", "deploy"),
    ("How long is telemetry available?", "logs"),
]
ALIASES = {"password": "credentials access keys", "archive": "backup snapshots",
           "payment": "billing invoices", "statement": "receipt invoice",
           "traffic": "api requests", "quota": "limited minute",
           "undo": "roll back", "rollout": "deployment release",
           "telemetry": "application logs"}
STOP = set("a an the i my do how when where can is are what after long should".split())
ACTIONS = ("literal", "expand aliases", "expand aliases and remove stop words")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve(question, action, k=2):
    words = tokens(question)
    if action > 0:
        words += [w for t in list(words) for w in tokens(ALIASES.get(t, ""))]
    if action == 2:
        words = [w for w in words if w not in STOP]
    query = set(words)
    scores = {key: len(query & set(tokens(value))) / max(1, len(query))
              for key, value in DOCUMENTS.items()}
    ranked = sorted(scores, key=lambda key: (-scores[key], key))[:k]
    return ranked, scores


def reward(question, gold, action):
    ranked, _ = retrieve(question, action)
    quality = 1.0 / (ranked.index(gold) + 1) if gold in ranked else 0.0
    return quality - (0.03 if action > 0 else 0.0)


if __name__ == "__main__":
    for action, name in enumerate(ACTIONS):
        ranks = [retrieve(q, action)[0] for q, _ in TEST]
        recall = sum(gold in rank for rank, (_, gold) in zip(ranks, TEST)) / len(TEST)
        mrr = sum(1 / (rank.index(gold) + 1) if gold in rank else 0
                  for rank, (_, gold) in zip(ranks, TEST)) / len(TEST)
        print(f"{name}: recall@2={recall:.3f}, MRR@2={mrr:.3f}")
