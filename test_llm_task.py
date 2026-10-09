import unittest

from llm_task import completion_text, prompt_for, score_query


class RetrievalRewardTests(unittest.TestCase):
    def test_relevant_query_beats_unrelated_query(self):
        self.assertEqual(score_query("backup snapshots restore", "backup")["reward"], 1.0)
        self.assertEqual(score_query("invoices billing receipt", "backup")["reward"], 0.0)

    def test_no_credit_for_zero_overlap_ties(self):
        self.assertEqual(score_query("xyzzy", "access")["reward"], 0.0)
        self.assertEqual(score_query("xyzzy", "backup")["ranked"], [])

    def test_invalid_queries_are_penalized(self):
        for query in ("", "backup\nsnapshots", "backup " * 13):
            self.assertEqual(score_query(query, "backup")["reward"], -1.0)

    def test_second_place_reward(self):
        self.assertEqual(score_query("console", "backup")["reward"], 0.5)

    def test_chat_completion_and_label_separation(self):
        self.assertEqual(completion_text([{"role": "assistant", "content": "backup"}]), "backup")
        prompt = prompt_for("Can I recover an archive?")
        self.assertEqual(prompt[-1]["content"], "Can I recover an archive?")
        self.assertNotIn("gold", str(prompt))


if __name__ == "__main__":
    unittest.main()
