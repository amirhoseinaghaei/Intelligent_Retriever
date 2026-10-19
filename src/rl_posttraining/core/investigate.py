"""Prompts and request validation for investigate query rewriting."""
import math


SYSTEM = (
    "Rewrite the user's software investigation request for ArtifactNet retrieval. "
    "Preserve the problem, requested behavior, constraints, and useful technical terms. "
    "Make the request precise enough to find relevant specifications, designs, or tests. "
    "Return only the rewritten request on one line. Do not answer it, invent facts, "
    "or include explanations, JSON, or tool names."
)


def prompt_for(question):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]


def arguments_for(record, text=None):
    """Normalize archived MCP arguments to the ArtifactNet backend API."""
    original = record["x"]
    question = original.get("changeRequest") or original.get("text")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Investigate input requires changeRequest or text")
    arguments = {"text": question.strip() if text is None else text,
                 "entryNodes": original.get("entryNodes", "spec"),
                 "kTop": original.get("kTop", 4), "hops": original.get("hops", 2),
                 "threshold": original.get("threshold", 0.3)}
    if not isinstance(arguments["entryNodes"], str) or not arguments["entryNodes"].strip():
        raise ValueError("entryNodes must be a nonempty label")
    for name, minimum in (("kTop", 1), ("hops", 0)):
        value = arguments[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    threshold = arguments["threshold"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")
    return arguments


def valid_query(query):
    return isinstance(query, str) and 0 < len(query.strip()) <= 2000 and "\n" not in query and "\r" not in query
