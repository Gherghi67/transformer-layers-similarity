# Transformer layers and attention-head similarity

Course project for LLMs & Cognition 2026, University of Bucharest.

Team: Bogdan Gherghina, Pănoiu Silviu-Cristian, Sîrbu Iulia-Georgiana.

We build per-task similarity maps of the layers and attention heads of small frozen models. It extends *Transformer Layers as Painters* to single tasks, to heads, to BERT and to newer attention designs.

## Base paper

Sun, Pickett, Nain, Jones, *Transformer Layers as Painters*, AAAI 2025 ([arXiv 2407.09298](https://arxiv.org/abs/2407.09298), [code](https://github.com/floatingbigcat/transformer_layers_as_painters)).

The authors perturb frozen Llama2 and BERT-Large: they skip, reorder, repeat or parallelize middle layers. They also compute the cosine similarity between layer hidden states.

What it found:
- Three layer groups: the first layer, a large uniform middle, and the last 1–2 layers.
- Middle layers share a representation space but are not redundant.
- GSM8K and ARC are the most order-sensitive tasks.

Gaps we work on:
- Its similarity maps are averaged over all tasks.
- BERT has no similarity map, only accuracy (App. A.9).
- There is no head-level analysis.

How their code changes layer order: [`gpt/routing_llama.py`](https://github.com/floatingbigcat/transformer_layers_as_painters/blob/main/gpt/routing_llama.py) patches `LlamaModel.forward` to loop over a list of layer indices, so skip, reverse, random, middle_repeat and loop_parallel are just different lists. Reordering works without fine-tuning because every layer maps tokens × d to tokens × d.

## Related work

| Paper | Takeaway for us |
| --- | --- |
| Kornblith et al. 2019 | CKA, the standard measure for comparing layer representations |
| Lad, Gurnee, Tegmark 2024 | Deleting or swapping layers; proposes "stages of inference" |
| Gromov et al. 2024, ShortGPT (Men et al. 2024) | Deep layers are redundant and can be pruned using inter-layer similarity |
| Godey et al. 2024, Sun et al. 2024 (Massive Activations) | Anisotropy and outlier dimensions inflate cosine similarity, which is why we also use CKA |
| Clark et al. 2019 | BERT heads cluster by behaviour; heads in the same layer often behave alike |
| Voita et al. 2019, Michel et al. 2019 | A few specialized heads do most of the work; many can be pruned |
| Pythia (Biderman et al. 2023) | Same data, many sizes; used in the base paper's appendix |

## Data

The base paper's benchmarks, all public on Hugging Face. The decoder maps only need frozen forward passes, so 300–500 examples per task are enough for stable similarity matrices.

| Task | Type | Split | Size | Models |
| --- | --- | --- | --- | --- |
| GSM8K | math reasoning | test | 1,319 | all decoder models |
| ARC-Challenge | science reasoning | test | 1,172 | all decoder models |
| HellaSwag | commonsense (semantic) | validation | 10,042 | all decoder models |
| WinoGrande | coreference (semantic) | validation | 1,267 | all decoder models |
| LAMBADA | next-word prediction | test | 5,153 | all decoder models |
| GLUE: CoLA, MRPC, QNLI, RTE, SST-2, STS-B, WNLI | classification / similarity | validation (train for fine-tuning) | 71 (WNLI) – 5.5k (QNLI) | BERT, ModernBERT |

The decoder models are Pythia, Qwen2.5, Gemma 3 and Qwen3.5.

## Method

All runs are on Google Colab with one 12 GB GPU, so smaller models stand in for Llama2. The only training is BERT fine-tuning.

- **Layers:** take the output of every transformer block (hidden states) and pool over tokens: the mean with padding masked, or [CLS] for BERT. This gives one L×L similarity matrix per task.
- **Heads:** forward hooks collect each head's attention pattern and its output before the output projection. This gives one (L·H)×(L·H) matrix per task.
- **BERT:** maps for the pretrained model vs. the model fine-tuned on each GLUE task.
- **Reasoning tasks:** we use every example, answered correctly or not, and also compare the maps for correct and incorrect answers.

Attention designs compared, with the same layer and head maps for each:

| Attention design | Model |
| --- | --- |
| Standard multi-head | Pythia 70M–1.4B |
| Grouped-query | Qwen2.5-1.5B |
| Interleaved local sliding-window / global | Gemma 3 1B; ModernBERT on the encoder side |
| Hybrid linear + full attention | Qwen3.5-0.8B (Gated DeltaNet) |
| Parallel attention + Mamba heads (optional) | Falcon-H1-0.5B |

We confirm the exact checkpoints once we have checked that they fit in memory.

Metrics:
- Similarity: cosine (as in the paper), mean-centered cosine, linear CKA; Jensen–Shannon divergence for attention patterns.
- Comparing maps: difference maps (reasoning minus semantic), correlation between maps, and the size and boundaries of the middle block, found by clustering the matrix.
- Sanity check: each model's task accuracy. We only compare maps where the model scores above chance.

## Planned comparisons

No results yet. The first result is a reproduction of the paper's Pythia-410M cosine map on LAMBADA (App. A.3, Fig. 14), to check the pipeline. Then comes the same map with CKA.

| Comparison | Judged by | What we expect |
| --- | --- | --- |
| Reasoning vs. semantic tasks | Difference maps, middle-block size | A smaller or more fragmented middle block on GSM8K/ARC |
| Cosine vs. CKA | Map correlation | CKA shows more structure inside the middle block |
| Model size (70M → 1.4B) | Map shape | Diagonal band at small sizes, block structure from ~410M (as in Fig. 14) |
| BERT pretrained vs. fine-tuned | Per-layer change in similarity | Fine-tuning mostly changes the top layers |
| Heads | Clusters of heads | Head groups that span several middle layers |
| Attention architectures | Middle-block size, head clusters per model | Local/global and linear/full-attention layers form distinct similarity groups |
| Correct vs. incorrect answers | Difference maps | Open question: does map structure relate to success on reasoning? |

## Risks

| Bottleneck | Fix |
| --- | --- |
| Small Pythia models score near chance on GSM8K and ARC-Challenge, so their "reasoning map" may not reflect reasoning | Qwen2.5-1.5B is well above chance on GSM8K and is already in the architecture comparison |
| Stored attention patterns grow with the square of sequence length | Sample examples and truncate long inputs |
| Token pooling and input length may drive task differences more than the task itself | Length-matched subsets; pooling is question 2 below |

## Questions for the instructor

These were sent with the preliminary report. We haven't had answers yet.

1. Is Pythia an acceptable substitute for Llama2, or should we include a small Llama-family model?
2. Which token pooling do you recommend: mean over tokens, or last token / [CLS]?
3. Are there attention architectures you would especially like us to include?
4. Should we also reproduce the paper's layer experiments (skipping or reordering layers and measuring how accuracy drops), or is the similarity analysis enough?

## Deadlines

Each progress report is worth 0.5p and is due at 10:00. We write them ourselves, not with an LLM (see the [course doc](https://docs.google.com/document/d/1UVS7hEYwRv2xeLq87nkmQ0DYRMKLrR20aRYtQhHxJyU/edit?tab=t.0)). We skip the 19 Nov slot.

| Report | Date |
| --- | --- |
| Progress report 1 | 03 Dec 2026 |
| Progress report 2 | 17 Dec 2026 |
| Progress report 3 | 14 Jan 2027 |
| Progress report 4 | 24 Jan 2027 |

Progress report 1 should show:
- the pipeline working, with the Pythia-410M LAMBADA map reproduced;
- the data and the EDA;
- layer maps per task and model;
- cosine vs. CKA;
- the first head maps;
- the accuracy check;
- BERT started.

## Tasks to 03 Dec

We do these as a team, with no fixed owners.

Setup (20–24 Nov)
- [ ] 1. Colab notebook in this repo; load Pythia-410M and collect every layer's hidden states (21 Nov)
- [ ] 2. Loaders for GSM8K, ARC-Challenge, HellaSwag, WinoGrande, LAMBADA and GLUE; a fixed sample of 300–500 per task (22 Nov)
- [ ] 3. Fit check on the 12 GB GPU: Qwen2.5-1.5B, Gemma 3 1B, Qwen3.5-0.8B, ModernBERT (Falcon-H1-0.5B optional) (22 Nov)
- [ ] 4. Reproduce Fig. 14: Pythia-410M cosine map on LAMBADA (24 Nov)
- [ ] 5. EDA: token lengths per task, share truncated, GLUE class balance (24 Nov)

Results and report (25 Nov – 03 Dec)
- [ ] 6. Same LAMBADA map with linear CKA and mean-centered cosine (26 Nov)
- [ ] 7. Accuracy of each model on each task; mark where it is above chance (27 Nov)
- [ ] 8. Layer maps for all 5 decoder tasks across Pythia 70M–1.4B; difference maps, reasoning minus semantic (29 Nov)
- [ ] 9. Head maps for Pythia-410M: attention patterns via hooks, Jensen–Shannon, (L·H)×(L·H) matrix (29 Nov)
- [ ] 10. Layer maps for Qwen2.5-1.5B, Gemma 3 1B and Qwen3.5-0.8B on the same tasks (30 Nov)
- [ ] 11. BERT-base pretrained maps on GLUE; fine-tune on SST-2 and RTE and redo the maps (30 Nov)
- [ ] 12. Write progress report 1 with figures, by hand; submit before 10:00 on 03 Dec (02 Dec)

## Open questions

- When do we meet each week to go through the tasks?
- Who has a GPU runtime?
