import json
import unittest

from benchmarks.loaders import SAMPLES_FILE, SEED, SAMPLE_SIZE, TASKS, format_row, pick


class FormatTest(unittest.TestCase):
    def test_gsm8k_keeps_final_number(self):
        row = {"question": "How many?", "answer": "3 * 400 = 1,200\n#### 1,200"}
        self.assertEqual(format_row("gsm8k", row), {"prompt": "Question: How many?\nAnswer:", "answer": "1200"})

    def test_arc_maps_answer_key_to_index(self):
        row = {"question": "Q?", "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]}, "answerKey": "B"}
        ex = format_row("arc_challenge", row)
        self.assertEqual((ex["label"], ex["answer"]), (1, "b"))

    def test_hellaswag_cleans_wikihow_tags(self):
        row = {"activity_label": "Home", "ctx_a": "Pick a pan. [title] Heat it.", "ctx_b": "you",
               "endings": ["[step] add oil.", "wait."], "label": "0"}
        ex = format_row("hellaswag", row)
        self.assertEqual(ex["prompt"], "Home: Pick a pan.. Heat it. You")
        self.assertEqual(ex["choices"], [" add oil.", "wait."])

    def test_winogrande_label_is_zero_based(self):
        row = {"sentence": "A beat B so _ won.", "option1": "A", "option2": "B", "answer": "1"}
        self.assertEqual(format_row("winogrande", row)["answer"], "A")

    def test_lambada_drops_last_word(self):
        ex = format_row("lambada", {"text": "he opened the door"})
        self.assertEqual(ex, {"prompt": "he opened the", "answer": "door"})

    def test_glue_single_sentence_has_no_second_text(self):
        ex = format_row("sst2", {"sentence": "good", "label": 1, "idx": 0})
        self.assertEqual(ex, {"text_a": "good", "text_b": None, "label": 1})


class SamplesTest(unittest.TestCase):
    def test_pick_is_fixed_and_sorted(self):
        self.assertEqual(pick(1000), pick(1000))
        self.assertEqual(pick(1000), sorted(pick(1000)))
        self.assertEqual(pick(71), list(range(71)))

    def test_samples_file_matches_pick(self):
        data = json.loads(SAMPLES_FILE.read_text())
        self.assertEqual((data["seed"], data["size"]), (SEED, SAMPLE_SIZE))
        self.assertEqual(list(data["tasks"]), TASKS)
        for task, entry in data["tasks"].items():
            self.assertEqual(entry["idx"], pick(entry["rows"]), task)


if __name__ == "__main__":
    unittest.main()
