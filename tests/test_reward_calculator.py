import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from rl_posttraining.integrations.reward_calculator import judge_relevance


class JudgeTests(unittest.TestCase):
    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'})
    @patch('urllib.request.urlopen')
    def test_request_and_validated_response(self, urlopen):
        urlopen.return_value = io.BytesIO(json.dumps({'choices': [{'message': {
            'content': json.dumps({'score': 0.7, 'reason': 'Partial'})}}]}).encode())
        self.assertEqual(judge_relevance('issue', [{'text': 'evidence'}], 'test')['score'], 0.7)
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body['model'], 'test')
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertNotIn('test-secret', request.data.decode())
        self.assertNotIn('temperature', body)

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'})
    @patch('urllib.request.urlopen')
    def test_http_error_reports_provider_message_without_credentials(self, urlopen):
        body = json.dumps({'error': {'message': 'Invalid model test-secret'}}).encode()
        urlopen.side_effect = urllib.error.HTTPError('https://example.test', 400,
                                                    'Bad Request', {}, io.BytesIO(body))
        with self.assertRaises(RuntimeError) as error:
            judge_relevance('issue', [], 'invalid/model')
        self.assertIn('HTTP 400', str(error.exception))
        self.assertIn('Invalid model', str(error.exception))
        self.assertIn('invalid/model', str(error.exception))
        self.assertNotIn('test-secret', str(error.exception))
        self.assertEqual(urlopen.call_count, 1)

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'})
    @patch('urllib.request.urlopen')
    def test_non_json_http_error_keeps_status(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError('https://example.test', 400,
                                                    'Bad Request', {}, io.BytesIO(b'<html>error</html>'))
        with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
            judge_relevance('issue', [], 'test')

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'})
    @patch('urllib.request.urlopen')
    def test_invalid_verdict_fails(self, urlopen):
        for score in (True, 2, '0.8'):
            urlopen.return_value = io.BytesIO(json.dumps({'choices': [{'message': {
                'content': json.dumps({'score': score, 'reason': 'Bad'})}}]}).encode())
            with self.assertRaises(RuntimeError):
                judge_relevance('issue', [], 'test')



if __name__ == '__main__':
    unittest.main()
