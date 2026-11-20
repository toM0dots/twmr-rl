# Transformable Wheel Robot

## Demo

<img src="assets/robot_step.gif" alt="Robot climbing over a step" style="width: 400px; max-width: 100%;">

## MuJoCo Playground Training Script

```bash
./.venv/bin/python train_jax_ppo.py --env_name="TransformableWheelMobileRobot"
```

This script is part of [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground). We should occasionally look for updates to the script. When we update the script we need to add `import twmr` to the top.

### TombotRecovery (recovery, driving, step climbing)

Environment: `packages/twmr/src/twmr/tombot_env.py`. Policy/critic: `sandbox/Tom/tombot_gnn.py`. Model: `packages/twmr/assets/tombot.xml`.

The task is to (1) recover to a flat, upright pose from a random (often tipped-over) start, (2) drive forward steadily while level, and (3) use the arms only when needed, i.e. while tipped over or when blocked by a step. The policy is blind (no forward sensor): it sees IMU, motor currents, wheel/arm encoders, its previous action and a 0.1/0.2/0.4 s history of those. Tuned PPO settings are in `tombot_ppo_config()` in `train_jax_ppo.py`.

Train in two stages. Stage A, recovery and flat driving:

```bash
./.venv/bin/python train_jax_ppo.py --env_name=TombotRecovery --suffix=stageA \
  --playground_config_overrides='{"obstacle_probability": 0.0}' --num_timesteps=100000000
```

Stage B, add steps (1-3.5 cm by default), continuing from stage A:

```bash
./.venv/bin/python train_jax_ppo.py --env_name=TombotRecovery --suffix=stageB \
  --load_checkpoint_path=logs/TombotRecovery-<stageA-timestamp>-stageA/checkpoints --num_timesteps=200000000
```

Things to watch (`logs/<run>/training_metrics.jsonl` or TensorBoard): `training/kl_mean` should stay around 0.01-0.03; `eval/episode_task/obstacle_clear_event` is the fraction of eval episodes that got past a step (compare with the obstacle probability); `eval/episode_reward/speed` is driving progress while level. Eval is deterministic by default (`--deterministic_eval`). Raise `obstacle_height_max` with `--playground_config_overrides` once the policy clears the current range.

Notes on the model: every step of 1 cm or more blocks the passive middle wheel, so the arms are always needed to climb. MJX's JAX backend has no cylinder-box collision, so each wheel carries an invisible sphere of the same radius that only touches the obstacle; the cylinders still handle the floor.

## Development Workflow

### Prerequisites

- Your system must have git installed (this is already installed on the HPC).
- You should install [Pixi](https://pixi.sh/latest/).
- Install [gh](https://cli.github.com/) with Pixi: `pixi global install gh`

Although a bit confusing, we'll be using both Pixi and uv. You should think of Pixi as a package manager for any tools we'll need, and uv as the Python-only package/workspace manager. We'll mostly use Pixi to install uv, and then use uv to manage our Python packages.

### Git One-Time Setup

```bash
# This only needs to be done once and you've likely done it already
git config --global user.name "YOUR NAME"
git config --global user.email "YOU@EXAMPLE.COM"

# Authenticate Pixi
gh auth login
```

### Initial Workspace Setup

> [!NOTE]
> You shouldn't have to do any of this if you clone the repository. These are notes for the from-scratch setup process that I used.

<details>

<summary>How to create the initial `uv` workspace.</summary>

These commands only need to be run once to setup the project.

```bash
# Assuming the directory was already created or cloned
pixi init .
pixi add uv

# Create a virtual environment with Python 3.11 and create the twmr package
# This will create a uv workspace and virtual environment
# --bare prevents creation of README.md, src/, .python-version, etc. (good for top level)
pixi run uv init --python 3.11 --bare
pixi run uv init --package packages/twmr

# Install playground dependencies and then playground
# First grab tool.uv.indexs and tool.uv.sources from mujoco_playground pyproject.toml
pixi run uv add "jax[cuda12]"
# --> test: .venv/bin/python -c "import jax; print(jax.default_backend())" --> gpu
pixi run uv add warp-lang
pixi run uv pip install "git+https://github.com/google-deepmind/mujoco_playground.git"
# --> test: .venv/bin/python -c "import mujoco_playground" --> no warnings
# Now manually add additional packages to the pyproject.toml file
```

</details>

### Initial Fork and Clone Process

This is the process you should use when you clone the repository for the first time on a new machine (i.e., the HPC/OnDemand).

```bash
# 1. Fork and clone the main repository (say yes to cloning)
cd to/a/resonable/directory
gh repo fork anthonyjclark/twmr-rl

# 2. Setup the project
cd twmr-rl
pixi run uv sync
source .venv/bin/activate
python -m ipykernel install --name mp --display-name "MuJoCo Playground" --user
```

Now you're ready to go!

### Subsequent Cloning Process

If you've already forked the main repository (`anthonyjclark/twmr-rl`) and just want to clone it on a new machine, you should not fork.

```bash
# 1. Clone your forked repository (say yes to cloning)
cd to/a/resonable/directory
gh repo clone YOUR_USERNAME/twmr-rl

# 2. Setup the project (same as the instructions for "Initial Fork and Clone Process")
```

### Pull Request Workflow

I recommend creating a branch when you are working on a large change (a new feature, a bug fix, etc).
This keeps your changes isolated from the main branch that we all use until they are well tested and ready to be merged in.
For smaller changes, you can just commit directly to your main branch and then open a pull request from your main branch to the main branch of the main repository.

You can use the command line to create a branch, commit your changes, and push the branch to your forked repository.
You can also use the VSCode interface if you prefer.

```bash
# 1. Create and switch to a new branch named
git checkout -b feat/SHORT-DESCRIPTIVE-NAME

# 2. Make changes to the code and then (repeat as is useful)
git add FILES_YOU_CHANGED/ADDED
git commit -m "SHORT DESCRIPTIVE MESSAGE"

# 3. Push branch to remote repository
git push -u origin feat/SHORT-DESCRIPTIVE-NAME

# Continue working on the feature branch until ready to merge
# Open a Pull Request (PR) on GitHub targeting main and then send a message to have it reviewed
# https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
#
# Alternatively, you can use the gh cli to create a pull request from the command line: https://cli.github.com/manual/gh_pr_create
```

### Upgrading a Dependency

We'll occasionally get `Failed to fetch` errors because an older version of a pinned depenency is no longer available (2026-02-04: this happened with our MuJoCo dependency). This is someone hard to debug because some people will still be able to install the dependency from cache.

```bash
# Recreate the dependency error by ignoring cache
pixi run uv --no-cache sync --upgrade-package mujoco

# Upgrade the dependency
pixi run uv sync --upgrade-package mujoco
git add
```
