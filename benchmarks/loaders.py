"""Loaders for the project's benchmarks and the fixed sample of each one.

Decoder tasks give dicts with idx, prompt and answer (plus choices and label
for multiple choice). The prompt is written the way lm-eval-harness writes it,
since the base paper ran its benchmarks through lm-eval. GLUE tasks give idx,
text_a, text_b (None for single-sentence tasks) and label, for BERT.

    from benchmarks import load_sample
    examples = load_sample("arc_challenge")

Regenerate samples.json (only needed if the sample size or seed changes):

    python -m benchmarks.loaders --force
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 0
SAMPLE_SIZE = 500
SAMPLES_FILE = Path(__file__).with_name("samples.json")

SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    "lambada": ("EleutherAI/lambada_openai", "default", "test"),
    "cola": ("nyu-mll/glue", "cola", "validation"),
    "mrpc": ("nyu-mll/glue", "mrpc", "validation"),
    "qnli": ("nyu-mll/glue", "qnli", "validation"),
    "rte": ("nyu-mll/glue", "rte", "validation"),
    "sst2": ("nyu-mll/glue", "sst2", "validation"),
    "stsb": ("nyu-mll/glue", "stsb", "validation"),
    "wnli": ("nyu-mll/glue", "wnli", "validation"),
}

GLUE_FIELDS = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASKS = list(SOURCES)
DECODER_TASKS = [t for t in TASKS if t not in GLUE_FIELDS]
GLUE_TASKS = list(GLUE_FIELDS)


def format_gsm8k(row):
    return {
        "prompt": f"Question: {row['question']}\nAnswer:",
        "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
    }


def format_arc(row):
    choices = row["choices"]["text"]
    label = row["choices"]["label"].index(row["answerKey"])
    return {
        "prompt": f"Question: {row['question']}\nAnswer:",
        "choices": choices,
        "label": label,
        "answer": choices[label],
    }


def clean_hellaswag(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def format_hellaswag(row):
    context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    choices = [clean_hellaswag(e) for e in row["endings"]]
    label = int(row["label"])
    return {
        "prompt": clean_hellaswag(row["activity_label"] + ": " + context),
        "choices": choices,
        "label": label,
        "answer": choices[label],
    }


def format_winogrande(row):
    choices = [row["option1"], row["option2"]]
    label = int(row["answer"]) - 1
    return {
        "prompt": row["sentence"],
        "choices": choices,
        "label": label,
        "answer": choices[label],
    }


def format_lambada(row):
    words = row["text"].split(" ")
    return {"prompt": " ".join(words[:-1]), "answer": words[-1]}


DECODER_FORMATS = {
    "gsm8k": format_gsm8k,
    "arc_challenge": format_arc,
    "hellaswag": format_hellaswag,
    "winogrande": format_winogrande,
    "lambada": format_lambada,
}


def format_row(task, row):
    if task in DECODER_FORMATS:
        return DECODER_FORMATS[task](row)
    a, b = GLUE_FIELDS[task]
    return {"text_a": row[a], "text_b": row[b] if b else None, "label": row["label"]}


def load_split(task):
    path, config, split = SOURCES[task]
    return load_dataset(path, config, split=split)


def load_all(task):
    return [{"idx": i, **format_row(task, row)} for i, row in enumerate(load_split(task))]


def load_sample(task):
    entry = json.loads(SAMPLES_FILE.read_text())["tasks"][task]
    ds = load_split(task)
    if len(ds) != entry["rows"]:
        raise ValueError(f"{task}: split has {len(ds)} rows, samples.json expects {entry['rows']}")
    return [{"idx": i, **format_row(task, ds[i])} for i in entry["idx"]]


def pick(rows, size=SAMPLE_SIZE, seed=SEED):
    if rows <= size:
        return list(range(rows))
    return sorted(random.Random(seed).sample(range(rows), size))


def make_samples(force=False):
    if SAMPLES_FILE.exists() and not force:
        raise SystemExit(f"{SAMPLES_FILE} exists; pass --force to overwrite it")
    tasks = {}
    for task in TASKS:
        path, config, split = SOURCES[task]
        rows = len(load_split(task))
        tasks[task] = {"dataset": path, "config": config, "split": split, "rows": rows, "idx": pick(rows)}
        print(f"{task}: {len(tasks[task]['idx'])} of {rows}")
    lines = ",\n".join(f"  {json.dumps(t)}: {json.dumps(v)}" for t, v in tasks.items())
    SAMPLES_FILE.write_text(f'{{"seed": {SEED}, "size": {SAMPLE_SIZE}, "tasks": {{\n{lines}\n}}}}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    make_samples(parser.parse_args().force)
