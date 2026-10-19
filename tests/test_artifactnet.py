"""Reward only retrieved graph evidence; don't reward generated analysis text."""
import io
import json
import os
import urllib.error
import unittest
from unittest.mock import patch

from rl_posttraining.core.investigate import arguments_for, prompt_for
from rl_posttraining.integrations import artifactnet


class ArtifactNetTests(unittest.TestCase):
    record = {"x": {"changeRequest": "original task", "entryNodes": "Requirement", "kTop": 5}}

    def test_request_mapping_and_original_question_for_judge(self):
        node = {"key": "Requirement:one", "text": "actual graph evidence"}
        result = {"queries": [{"change": "invented analysis", "retrieval": {
            "changes": [{"nodes": [node, node]}]}}]}
        with patch.object(artifactnet, "investigate", return_value=result) as investigate, \
                patch.object(artifactnet, "judge_relevance", return_value={"score": 0.8}) as judge:
            scored = artifactnet.score_investigate("rewritten task", self.record, "http://local", "judge")
        self.assertEqual(scored["reward"], 0.8)
        self.assertEqual(investigate.call_args.args[1]["text"], "rewritten task")
        self.assertEqual(investigate.call_args.args[1]["entryNodes"], "Requirement")
        self.assertEqual(judge.call_args.args[:2], ("original task", [{"id": node["key"], "text": node["text"]}]))

    def test_empty_results_and_invalid_completions_do_not_call_judge(self):
        with patch.object(artifactnet, "investigate", return_value={"queries": []}) as investigate, \
                patch.object(artifactnet, "judge_relevance") as judge:
            self.assertEqual(artifactnet.score_investigate("query", self.record, "url", "judge")["reward"], 0)
            self.assertEqual(artifactnet.score_investigate("bad\nquery", self.record, "url", "judge")["reward"], -1)
            self.assertEqual(investigate.call_count, 1)
            judge.assert_not_called()

    def test_new_node_table_and_http_envelope(self):
        result = {"nodes": [{"key": "k", "text": "evidence"}], "queries": []}
        response = io.BytesIO(json.dumps({"sessionId": "s", "result": result}).encode())
        with patch("urllib.request.urlopen", return_value=response) as call:
            self.assertEqual(artifactnet.investigate("http://local/artifactnet", {"text": "task"}), result)
        self.assertEqual(call.call_args.args[0].full_url, "http://local/artifactnet/tools/call")
        self.assertEqual(artifactnet.evidence_documents(result), [{"id": "k", "text": "evidence"}])

    def test_api_failure_stops_instead_of_training_on_negative_reward(self):
        with patch.object(artifactnet, "investigate", side_effect=RuntimeError("service unavailable")):
            with self.assertRaisesRegex(RuntimeError, "service unavailable"):
                artifactnet.score_investigate("query", self.record, "url", "judge")

    def test_http_error_keeps_backend_explanation_without_credentials(self):
        body = json.dumps({'error': {'message': 'Provider rejected test-secret'}}).encode()
        error = urllib.error.HTTPError('http://local', 400, 'Bad Request', {}, io.BytesIO(body))
        with patch('urllib.request.urlopen', side_effect=error), \
                patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'}):
            with self.assertRaises(RuntimeError) as raised:
                artifactnet.investigate('http://local/artifactnet', {'text': 'query'})
        self.assertIn('HTTP 400', str(raised.exception))
        self.assertIn('Provider rejected', str(raised.exception))
        self.assertNotIn('test-secret', str(raised.exception))

    def test_non_json_http_error_keeps_status_and_body(self):
        error = urllib.error.HTTPError('http://local', 501, 'Unsupported', {}, io.BytesIO(b'unsupported operation'))
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 501: unsupported operation'):
                artifactnet.investigate('http://local/artifactnet', {'text': 'query'})

    def test_backend_fetch_failure_retries_identical_read_request(self):
        error = urllib.error.HTTPError('http://local', 400, 'Bad Request', {},
                                      io.BytesIO(b'{"error":{"message":"fetch failed"}}'))
        response = io.BytesIO(b'{"result":{"queries":[]}}')
        with patch('urllib.request.urlopen', side_effect=[error, response]) as request, \
                patch.object(artifactnet.time, 'sleep') as sleep:
            result = artifactnet.investigate('http://local/artifactnet', {'text': 'query'})
        self.assertEqual(result, {'queries': []})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[0].data, request.call_args_list[1].args[0].data)
        sleep.assert_called_once_with(2)

    def test_persistent_fetch_failure_stops_after_three_attempts(self):
        errors = [urllib.error.HTTPError('http://local', 400, 'Bad Request', {},
                                        io.BytesIO(b'{"error":{"message":"fetch failed"}}')) for _ in range(3)]
        with patch('urllib.request.urlopen', side_effect=errors) as request, \
                patch.object(artifactnet.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400: fetch failed'):
                artifactnet.investigate('http://local/artifactnet', {'text': 'query'})
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_internal_llm_tag_failure_retries_without_changing_candidate(self):
        message = ('[LLM] Invalid LLM schema tag extraction after one repair attempt: '
                   'missing tag result for selected schema entry: design/DecisionRecord')
        error = urllib.error.HTTPError('http://local', 400, 'Bad Request', {},
                                      io.BytesIO(json.dumps({'error': {'message': message}}).encode()))
        response = io.BytesIO(b'{"result":{"queries":[]}}')
        with patch('urllib.request.urlopen', side_effect=[error, response]) as request, \
                patch.object(artifactnet.time, 'sleep'):
            result = artifactnet.investigate('http://local/artifactnet', {'text': 'unchanged candidate'})
        self.assertEqual(result, {'queries': []})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[0].data, request.call_args_list[1].args[0].data)

    def test_persistent_llm_output_failure_stops_after_three_attempts(self):
        message = '[LLM] Invalid LLM schema tag extraction after one repair attempt: missing tag result'
        errors = [urllib.error.HTTPError('http://local', 400, 'Bad Request', {},
                                        io.BytesIO(json.dumps({'error': {'message': message}}).encode()))
                  for _ in range(3)]
        with patch('urllib.request.urlopen', side_effect=errors) as request, \
                patch.object(artifactnet.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'missing tag result'):
                artifactnet.investigate('http://local/artifactnet', {'text': 'candidate'})
        self.assertEqual(request.call_count, 3)

    def test_request_and_graph_schema_errors_are_not_retried(self):
        for message in ('entryNodes must be a string', 'Invalid KG schema: unsupported field',
                        'Neo4j authentication failed'):
            with self.subTest(message=message):
                error = urllib.error.HTTPError('http://local', 400, 'Bad Request', {},
                                              io.BytesIO(json.dumps({'error': {'message': message}}).encode()))
                with patch('urllib.request.urlopen', side_effect=error) as request, \
                        patch.object(artifactnet.time, 'sleep') as sleep:
                    with self.assertRaises(RuntimeError):
                        artifactnet.investigate('http://local/artifactnet', {'text': 'candidate'})
                self.assertEqual(request.call_count, 1)
                sleep.assert_not_called()

    def test_snapshot_prompt_has_no_synthetic_examples_or_reference_output(self):
        messages = prompt_for("software task")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["content"], "software task")
        self.assertEqual(arguments_for(self.record)["text"], "original task")


if __name__ == "__main__":
    unittest.main()
