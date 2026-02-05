# TODO

- Improve `data/ratio` (non-pad / total tokens) to reduce wasted GPU compute (currently ~0.45 in UltraChat runs).
  - Candidates (in increasing complexity): avoid power-of-two padding, length bucketing, token-budget batching, on-the-fly packing, max_length curriculum.
  - Decide based on profiling: input-bound (CPU/dataloader) vs compute-bound (GPU padding waste).

