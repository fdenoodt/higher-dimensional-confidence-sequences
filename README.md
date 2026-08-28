# On the Tightness and Computational Tractability of Higher-Dimensional Confidence Sequences

Official implementation of the confidence-sequence constructions and experiments from the paper **"On the Tightness and Computational Tractability of Higher-Dimensional Confidence Sequences"**, currently under review at NeurIPS 2026.

The current manuscript is available as a [PDF](paper.pdf).

The accompanying tutorial, [Monitoring ML Models with Confidence Sequences](https://fdenoodt.github.io/higher-dimensional-confidence-sequences/2026/04/30/monitoring-ml-models-with-confidence-sequences/), shows how to use the bounding-box construction to monitor overall and subgroup model performance.

## Installation

The experiments were run with Python 3.9. Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the experiments

Run the synthetic-data experiments with:

```bash
python run_experiments_async.py
```

Set `small_scale = True` near the bottom of `run_experiments_async.py` for the small-scale Bounded UP appendix experiment. Keep it `False` for the main-paper configuration. Results are written to `logs/`.

Additional applications:

- `python -m adaptive_sample.conf_sequences.ab_testing` runs the multi-metric A/B testing experiment.
- [`adaptive_sample/conf_sequences/model_comparison/README.md`](adaptive_sample/conf_sequences/model_comparison/README.md) documents the Adult dataset model-comparison experiment.
- `python examples/model_monitoring/tutorial.py` reproduces the figures used in the tutorial.

The primary experiment configuration and available methods are defined in `main.py` and `run_experiments_async.py`.

## Repository structure

- `adaptive_sample/`: confidence-sequence implementations and experiment code
- `shared/`: command-line and experiment utilities
- `examples/`: runnable code accompanying the tutorial
- `_posts/`, `_layouts/`, and `assets/`: GitHub Pages blog source

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{denoodt2026tightness,
  title  = {On the Tightness and Computational Tractability of Higher-Dimensional Confidence Sequences},
  author = {Denoodt, Fabian and Hess, Sibylle and Vanschoren, Joaquin and Naesseth, Christian A.},
  year   = {2026},
  note   = {Manuscript under review at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  url    = {https://github.com/fdenoodt/higher-dimensional-confidence-sequences/blob/main/paper.pdf}
}
```

## License

This project is released under the [MIT License](LICENSE).
