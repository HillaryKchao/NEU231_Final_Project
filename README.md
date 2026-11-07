# NEU231_Final_Project

To run the repo model --

python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

python base_model.py --dataset fmnist --dends 16 --soma 128 --sampling local

## V1-style dendritic nonlinearities

`--nonlin v1` replaces each dendrite's LeakyReLU with three fixed nonlinearities applied in
parallel (supralinear `relu(u)^p`, sublinear `s*tanh(u/s)`, calcium spike
`A*sigmoid(k*(u-theta))`). Each soma then learns a weighted sum over all three channels of
each of its own dendrites. The default `--nonlin leaky` is the paper's model, unchanged.

python base_model.py --nonlin v1 --dends 16 --soma 256            # defaults: p=2, s=1, theta=1, k=10, A=1
python base_model.py --nonlin v1 --ca-thresh 0.5 --supra-p 1.5    # change the fixed constants

Results are tagged `dann_v1_*`; `plot_results.py` draws them next to the baseline (dashed
lines, square markers). `analyze_v1.py results/dann_v1_... --data-dir data` reports which
channel each soma relies on and whether dendrites reach the Ca threshold.

## Control 1: homogeneous type ("any single nonlinearity is enough")

`--nonlin supra`, `--nonlin sub` and `--nonlin ca` give every dendrite just that one
nonlinearity, with the baseline's cable layer and parameter count. Together with the LeakyReLU
baseline they are the single-nonlinearity controls for the mixed `v1` model.

bash sweep_controls.sh                      # seeds 1-3 at 16x256 for leaky, supra, sub, ca, v1
SEEDS="1 2 3 4 5" SIZES="16x256 64x256" bash sweep_controls.sh
python plot_results.py                      # figures/7_control_homogeneous.png + control1_homogeneous.csv

The CSV compares the mixed model with the best single nonlinearity at each size and gives a
verdict: "mixed wins" if the gap exceeds two combined SDs (and 0.5 pp), else "single is enough".

### Ratio experiment: one nonlinearity per dendrite

`--nonlin branch --mix S:B:C` gives every dendrite a single nonlinearity, with the ratio
setting how many of each soma's dendrites are supra, sub and Ca (12:3:1 at 16 dendrites is
exactly 12, 3 and 1). The cable layer and parameter count are the baseline's, so this tests
mixing across branches without the extra cable weights of `v1`.

Default ratios: `12:3:1` (biological: pyramidal dendrites are mostly NMDA-spike branches,
sublinear integration is a minority regime, and the Ca spike is one apical site per cell),
`1:1:1` (uniform), `1:3:12` (Ca-heavy counterfactual).

NONLINS=branch bash sweep_controls.sh        # the three ratios, seeds 1-3 at 16x256 (~25 min on MPS)
python plot_results.py                       # figures/8_ratio_sweep.png + ratio_sweep.csv

Figure 8 shows each ratio with error bars against the best single nonlinearity and the V1
mixed model at the same size; the CSV adds a "ratio wins" / "no advantage" verdict.

## Tuning the biological nonlinearities

The fixed constants (Ca threshold, gain, exponent, saturation) were initial guesses, and the
single-nonlinearity controls lose about a point to LeakyReLU on CIFAR-10 partly because the
fixed functions clip: Ca is bounded and flat away from threshold, tanh saturates, relu^p is
zero for negative input. Two remedies:

- `--leak L` adds a passive component to every channel, `f(u) + L*u` (real dendrites conduct
  passively alongside any active mechanism; 0.1 is the slope LeakyReLU keeps). Default 0.
- `sweep_constants.sh` runs a grid over the constants (and the Ca gate below) with one seed.
  `python plot_results.py --select best-val` ranks the variants by **validation** accuracy,
  never test, prints the winners as grid lines, and uses them in the comparison figures. Rerun
  the winners with `SEEDS="2 3" GRID=<file>` before drawing conclusions.

Runs with non-default constants get a tag suffix (`..._thr0.5_leak0.1_seed1`) and appear in
the figures as `model [thr0.5_leak0.1]`. `9_constants_sweep.png` / `constants_sweep.csv` show
every variant.

## Ca gate model (`--nonlin gate`)

The Ca spike's job in a pyramidal cell is not to add to the somatic input but to detect
coincidence between apical and basal drive and turn a spike into a burst. The gate model gives
each soma that computation, which no weighted sum of fixed functions can express:

    B  = basal dendrites' drive        (12 of 16 dendrites, LeakyReLU by default)
    A  = apical dendrites' drive        (4 of 16)
    Ca = sigmoid(k (A - theta))         apical plateau in [0, 1]
    soma = LeakyReLU( B * (1 + g Ca) )  apical gates basal

Flags: `--apical-frac` (0.25), `--basal-nonlin {leaky,supra,sub,ca}` (leaky, so a win is
attributable to the gate), `--ca-thresh`, `--ca-gain`, `--ca-amp`. Masked weights equal the
baseline's; there is one extra bias vector. `analyze_v1.py` reports how often the gate opens
and the correlation between basal drive and gate (`10_gate_activity.png`). Note that theta is
applied to the learned apical sum, so the apical bias can shift the effective threshold.

## CIFAR-10 on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HillaryKchao/NEU231_Final_Project/blob/bio-tuning/colab_cifar10.ipynb)

CIFAR-10 is a single 160 MB archive and each epoch reads all of it, so training needs the
dataset on the machine doing the work. `colab_cifar10.ipynb` runs the whole comparison on a
free Colab GPU instead: the dataset lands on Colab's disk, not yours, and the sweep takes a
few hours rather than 10+ on a laptop.

1. Open the notebook with the badge, pick a GPU runtime, run the cells top to bottom.
2. It clones this branch, smoke-tests one epoch, runs control 1 and the ratio sweep with
   `DATASET=cifar10`, and makes the figures in `figures/cifar10/`.
3. The last cell downloads a zip of the result JSONs, figures and log. Unzip into `results/`
   here and run `python plot_results.py --dataset cifar10 --out figures/cifar10` to regenerate.

Free sessions can time out; the sweep skips finished runs, so re-running the cells resumes.
Set `USE_DRIVE = True` in the notebook to keep results in Google Drive across sessions.

`--train-subset N` trains on the first N shuffled training images (validation and test stay
full). It is for quick checks: such runs are tagged `_nN` and `plot_results.py` leaves them out.
