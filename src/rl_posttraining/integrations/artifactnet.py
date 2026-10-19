"""Execute investigate requests and reward actual graph evidence, not LLM prose."""
import json
import logging
import os
import time
import urllib.error
import urllib.request

from rl_posttraining.core.investigate import arguments_for, valid_query
from .reward_calculator import judge_relevance

LOGGER = logging.getLogger(__name__)
MAX_ATTEMPTS = 3


def http_error_message(error):
    """Keep backend/provider explanations while redacting runtime credentials."""
    detail = error.read(65536).decode('utf-8', errors='replace')
    try:
        body = json.loads(detail)
        message = body.get('error', body.get('message', detail))
        if isinstance(message, dict):
            message = message.get('message', message)
        detail = message if isinstance(message, str) else json.dumps(message)
    except (ValueError, AttributeError):
        pass
    for name, value in os.environ.items():
        if value and any(word in name.upper() for word in ('KEY', 'TOKEN', 'PASSWORD')):
            detail = detail.replace(value, '[REDACTED]')
    return f'ArtifactNet HTTP {error.code}: {detail[:3000] or error.reason}'


def retryable_backend_error(status, message):
    """Separate provider failures from invalid requests and graph/schema errors."""
    if status in {429, 502, 503, 504}:
        return True
    if status != 400:
        return False
    if message in {
        'ArtifactNet HTTP 400: fetch failed',
        'ArtifactNet HTTP 400: [LLM] API request failed: fetch failed',
    }:
        return True
    # ArtifactNet validates its own LLM output and tries one repair internally.
    # A failed repair is an environment generation failure, not a bad policy action.
    return (message.startswith('ArtifactNet HTTP 400: [LLM] Invalid LLM ')
            and ' after one repair attempt:' in message)


def investigate(endpoint, arguments, timeout=180):
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/tools/call",
        data=json.dumps({"tool": "artifactnet_investigate", "arguments": arguments}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            message = http_error_message(error)
            if not retryable_backend_error(error.code, message) or attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(message) from None
        except (urllib.error.URLError, TimeoutError):
            message = 'ArtifactNet connection failed or timed out'
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(message) from None
        delay = 2 ** (attempt + 1)
        LOGGER.warning('%s; retrying investigate in %ss (attempt %s/%s)',
                       message, delay, attempt + 2, MAX_ATTEMPTS)
        time.sleep(delay)
    if not isinstance(payload, dict) or "result" not in payload or "error" in payload:
        raise RuntimeError("ArtifactNet returned an invalid response")
    result = payload["result"]
    if not isinstance(result, dict):
        raise RuntimeError("ArtifactNet investigate result must be an object")
    return result


def evidence_documents(result):
    """Handle both archived nested retrievals and the newer shared node table."""
    if "nodes" in result:
        nodes = result["nodes"]
    elif isinstance(result.get("queries"), list):
        nodes = []
        for query in result["queries"]:
            retrieval = query.get("retrieval") or {}
            for change in retrieval.get("changes", []):
                nodes.extend(change.get("nodes", []))
    else:
        raise RuntimeError("Unsupported ArtifactNet investigate result format")
    documents = {}
    for node in nodes:
        text = node.get("text")
        if isinstance(text, str) and text.strip():
            key = node.get("key") or text
            documents[key] = {"id": key, "text": text}
    return list(documents.values())


def score_investigate(query, record, endpoint, judge_model, timeout=180):
    valid = valid_query(query)
    query = query.strip()
    result, documents = None, []
    if valid:
        try:
            result = investigate(endpoint, arguments_for(record, text=query), timeout)
        except RuntimeError as error:
            raise RuntimeError(f'{error}; candidate query: {query!r}') from None
        documents = evidence_documents(result)
    verdict = ({"score": 0.0, "reason": "Invalid query" if not valid else "No retrieved evidence",
                "model": judge_model} if not documents else
               judge_relevance(arguments_for(record)["text"], documents, judge_model))
    return {"query": query, "valid": valid, "retrieval": result,
            "retrieved_documents": documents, "judge": verdict,
            "reward": verdict["score"] if valid else -1.0}
