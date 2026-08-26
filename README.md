# Virgo-DeepClean

Analysis pipeline and scripts for applying **DeepClean**, a convolutional neural
network that uses auxiliary "witness" sensors to regress non-linear,
non-stationary noise out of gravitational-wave strain data — to **Virgo**
detector data (O3b), together with the full set of validation studies used to
assess its impact on astrophysical searches: amplitude spectral density (ASD)
comparisons, BNS inspiral range, matched-filter SNR recovery, injected-signal
parameter estimation (PE), and Kolmogorov–Smirnov (KS) consistency tests on PE
posteriors.

This repository accompanies the results published in Kiendrébéogo et al.
(2025), *Phys. Rev. D* **112**, 083053, see [Citation](#citation).

## Built on DeepClean-prod

The neural-network training/cleaning engine itself (`deepclean_prod`,
imported throughout this repo) is **not** part of this repository. It is the
production DeepClean pipeline developed for LIGO/Virgo/KAGRA noise
regression:

- Upstream repository: <https://git.ligo.org/tri.nguyen/deepclean-prod>
- Authors/maintainers: Tri Nguyen, Michael Coughlin, Rich Ormiston,
  Rana Adhikari, Gabriele Vajente, Erik Katsavounidis, Muhammed Saleem

This project consumes `deepclean_prod` as a dependency (see
[Installation](#installation)) and adds Virgo-specific configuration,
injection/PE studies, and post-hoc validation analyses on top of it.

### Related analysis

The low-latency demonstration of DeepClean on LIGO data, which motivated and
informed the Virgo study in this repository, is described in Saleem et al.
(2024), *Class. Quantum Grav.* **41**, 195024, see
[Citation](#citation). The parameter-estimation results from that companion
study are included here for comparison under
[`analysis/virgo_pe/full_param_pe/saleem`](analysis/virgo_pe/full_param_pe/saleem).

## Repository structure

| Path | Contents |
|---|---|
| [`config/`](config) | Condor/DAGMan configuration (`config_virgo.ini`), witness-channel lists (single-layer and multi-layer), and the DAGMan generation script used to launch training/cleaning jobs. |
| [`analysis/train-cleaning/`](analysis/train-cleaning) | Data download and DeepClean training/cleaning drivers (single- and multi-layer) for the 142–162 Hz band. |
| [`analysis/sequential-single-pass/`](analysis/sequential-single-pass) | Sequential (layer-by-layer) vs. simultaneous multi-witness training strategy, and per-layer witness-channel configs. |
| [`analysis/asd/`](analysis/asd) | Amplitude spectral density (ASD) and ASD-ratio plots comparing original vs. DeepClean-cleaned strain. |
| [`analysis/loss-function/`](analysis/loss-function) | Training-loss-vs-epoch data and plots. |
| [`analysis/injection/`](analysis/injection) | BBH injection priors and strain generation/injection utilities (Bilby-based) used to build the signal-recovery test set. |
| [`analysis/match-filter-snr/`](analysis/match-filter-snr) | Matched-filter SNR recovery before/after DeepClean across frequency bands. |
| [`analysis/bns-range/`](analysis/bns-range) | Binary neutron star inspiral-range estimation and comparison before/after cleaning. |
| [`analysis/virgo_pe/`](analysis/virgo_pe) | Full and single-parameter Bayesian PE (`dynesty`/Bilby) on injected signals, run on original, DeepClean-cleaned, and zero-noise data; includes the companion [`saleem`](analysis/virgo_pe/full_param_pe/saleem) PE analysis. |
| [`analysis/ks-test/`](analysis/ks-test) | Kolmogorov–Smirnov tests of PE credible levels (DeepClean vs. uncleaned, and vs. the uniform distribution expected for unbiased recovery). |
| [`paper-plots/`](paper-plots) | Final figures used in the paper. |
| [`scripts/`](scripts) | Environment setup and DeepClean-prod installation helpers. |

All validation studies (training, ASD, injections, SNR, BNS range, PE, KS
tests) live under [`analysis/`](analysis); `config/`, `paper-plots/`, and
`scripts/` stay at the repository root since they are not study-specific.

Each subdirectory's scripts carry their own header docstrings with a
description and example invocation (`--help` is also available on the
argparse-based ones).

## Installation

Requires Python ≥ 3.9. [`pyproject.toml`](pyproject.toml) is the single
source of truth for dependencies; pick either path below.

```bash
git clone https://github.com/weizmannk/Virgo-DeepClean.git
cd Virgo-DeepClean
```

**conda** (creates a `virgo-deepclean` env, then offers to fetch
deepclean-prod):

```bash
make setup   # or: scripts/setup_env.sh
```

Or manually:

```bash
conda create -n virgo-deepclean python=3.10
conda activate virgo-deepclean
pip install .
```

**uv** (no conda required — see [below](#faster-setup-with-uv)):

```bash
make setup-uv   # or: uv sync
```

See the [`Makefile`](Makefile) for the other shortcuts (`make
install-deepclean`, `make install-deepclean-cluster`, `make lint`, `make
clean`).

### Installing the original DeepClean-prod pipeline

`deepclean_prod` is not on PyPI, and its own `environment.yml`/
`requirements.txt` are a ~2020 snapshot (Python 3.7, CUDA 10.2,
`torch==1.0.0`) that only resolves on the exact platform it was built for.
`scripts/install_deepclean_prod.sh` clones
<https://git.ligo.org/tri.nguyen/deepclean-prod> into
`config/virgo-deepclean-prod/` (cloning requires a LIGO.ORG account) and
supports two install modes:

```bash
# On a LIGO/Virgo Linux cluster: recreate deepclean-prod's own conda env
# from environment.yml and `pip install .` inside it, exactly as upstream
# documents. Works there because the platform (linux-64 + Python 3.7 +
# CUDA 10.2) matches what the pins were built for.
scripts/install_deepclean_prod.sh --mode cluster

# On a local machine / laptop (e.g. macOS): the frozen 2020 conda/CUDA
# stack won't resolve, so this installs a modern torch into your current
# environment, then `pip install -e . --no-deps` to avoid the stale
# torch==1.0.0 pin. This is the default mode.
scripts/install_deepclean_prod.sh --mode lite
```

If you already have a local copy (e.g. a `deepclean-prod.zip` archive), pass
its path instead of letting the script clone it:

```bash
scripts/install_deepclean_prod.sh --mode lite /path/to/deepclean-prod.zip
```

### Faster setup with uv

[uv](https://docs.astral.sh/uv/) reads [`pyproject.toml`](pyproject.toml)
and creates `.venv/` with all dependencies resolved and locked
(`uv.lock`), no conda required:

```bash
uv sync
```

`lalsuite` and `pycbc` do ship PyPI wheels for common platforms, so this
works on most machines, but conda remains the more battle-tested route on
LIGO/Virgo clusters. `deepclean_prod` still needs the separate install step
above — activate `.venv` first (`source .venv/bin/activate`), then run
`scripts/install_deepclean_prod.sh` as usual.

### Development

```bash
pip install pre-commit
pre-commit install
```

## Usage

Most scripts are standalone and self-documented; run with `--help` or read
the header docstring for the exact CLI. Typical workflow:

1. Configure witness channels and Condor job parameters in
   [`config/config_virgo.ini`](config/config_virgo.ini) and
   [`config/witnesses/`](config/witnesses).
2. Generate and submit the training/cleaning DAG:
   `python config/dagman_generate_v3.py config/config_virgo.ini --submit`.
3. Once cleaned frames are produced, run the validation analyses
   (ASD ratio, BNS range, matched-filter SNR, injections + PE, KS tests) from
   their respective directories, pointing `--path-dir`/`--outdir` at your
   output frames.

## Citation

If you use this code, please cite the paper this repository accompanies:

```bibtex
@article{Kiendrebeogo_2025,
  title     = {Application of nonlinear noise regression in the Virgo detector},
  author    = {Kiendrebeogo, R. Weizmann and Saleem, Muhammed and Bizouard, Marie Anne
               and Chen, Andy H. Y. and Christensen, Nelson and Chou, Chia-Jui
               and Coughlin, Michael W. and Janssens, Kamiel and Kam, S. Zacharie
               and Koulidiati, Jean and Yeh, Shu-Wei},
  journal   = {Phys. Rev. D},
  volume    = {112},
  issue     = {8},
  pages     = {083053},
  numpages  = {18},
  year      = {2025},
  month     = {Oct},
  publisher = {American Physical Society},
  doi       = {10.1103/hwnc-r7yy},
  url       = {https://link.aps.org/doi/10.1103/hwnc-r7yy}
}
```

and, for the DeepClean low-latency demonstration this work builds on:

```bibtex
@article{Saleem_2024,
  doi       = {10.1088/1361-6382/ad708a},
  url       = {https://doi.org/10.1088/1361-6382/ad708a},
  year      = {2024},
  month     = {sep},
  publisher = {IOP Publishing},
  volume    = {41},
  number    = {19},
  pages     = {195024},
  author    = {Saleem, Muhammed and Gunny, Alec and Chou, Chia-Jui and Yang, Li-Cheng
               and Yeh, Shu-Wei and Chen, Andy H Y and Magee, Ryan and Benoit, William
               and Nguyen, Tri and Fan, Pinchen and Chatterjee, Deep and Marx, Ethan
               and Moreno, Eric and Omer, Rafia and Raikman, Ryan and Rankin, Dylan
               and Sharma, Ritwik and Coughlin, Michael and Harris, Philip
               and Katsavounidis, Erik},
  title     = {Demonstration of machine learning-assisted low-latency noise
               regression in gravitational wave detectors},
  journal   = {Classical and Quantum Gravity}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

Distributed under the GNU Affero General Public License v3.0 — see
[LICENSE](LICENSE).

## Contact

Ramodgwendé Weizmann Kiendrébéogo — weizmann.kiendrebeogo@oca.eu
