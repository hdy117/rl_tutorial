# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

- Install dependencies: `pip install -r requirements.txt`
- Run the introductory CartPole walkthrough: `python 01_intro.py`
- Run the random-agent baseline and write `random_scores.npy`: `python 02_random_agent.py`
- Train the tabular Q-learning agent and write `q_learning_scores.npy` plus `q_table.npy`: `python 03_q_learning.py`
- Plot saved results and write `training_curve.png`: `python plot_results.py`
- Re-run the full experiment pipeline from the repo root: `python 02_random_agent.py && python 03_q_learning.py && python plot_results.py`

## Validation

- There is currently no formal test suite, linter, or build system configured in this repository.
- Validate changes by re-running the affected script from the repository root.
- For end-to-end changes, use the full pipeline command above so the `.npy` artifacts and plot are regenerated together.
- There is no single-test command yet because no test framework is configured.

## Architecture overview

- This is a flat, script-driven Python tutorial project, not a packaged library. The repository is organized around runnable lesson scripts plus one long teaching document.
- `RL_Tutorial.md` is the main narrative/tutorial document. It explains the RL concepts and roadmap, while the Python scripts provide concrete runnable examples.
- `01_intro.py` is the environment-inspection step: it creates `CartPole-v1`, prints the action/state meaning, and manually steps through the environment to illustrate state, action, reward, and episode termination.
- `02_random_agent.py` is the baseline experiment: it runs many episodes with random actions, reports score statistics, and saves `random_scores.npy` for later comparison.
- `03_q_learning.py` is the main learning script: it discretizes the continuous CartPole observation into buckets, uses an epsilon-greedy policy, updates a tabular Q-table with the Bellman target, and saves both `q_learning_scores.npy` and `q_table.npy`.
- `plot_results.py` is the reporting layer: it loads the saved `.npy` artifacts, plots the training curve and baseline-vs-trained comparison, and saves `training_curve.png`.

## Working assumptions in this repo

- Scripts communicate through files in the repository root rather than through shared Python modules. Run commands from the repo root so reads/writes to `.npy` files resolve correctly.
- The numbered scripts form a teaching sequence. If you extend the tutorial, preserve that staged flow unless you are intentionally refactoring the project structure.
- `RL_Tutorial.md` and `03_q_learning.py` are already coupled conceptually: the markdown explains the Bellman/Q-table logic that the script implements. When changing the algorithm description, variable naming, or learning flow, check both files.
- `01_intro.py` and `02_random_agent.py` create CartPole with `sutton_barto_reward=True`, while `03_q_learning.py` uses the default CartPole reward behavior. If you change scoring, evaluation, or comparisons, verify that the reward assumptions remain consistent across scripts.
- Files such as `random_scores.npy`, `q_learning_scores.npy`, `q_table.npy`, and `training_curve.png` are generated outputs, not primary source files.
