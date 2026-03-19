# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

- Install Python dependencies used by the tutorial tooling: `pip install -r requirements.txt`
- Regenerate the merged tutorial document from the repo root: `python merge_tutorial.py`
- Build the Bellman DAG example: `cd 3.0_bellman_cpp_cheese_example && ./build.sh`
- Run the Bellman DAG example: `./3.0_bellman_cpp_cheese_example/build/bellman_update_cheese_example`
- Build the Q-learning C++ example: `cd 4.0_q_learning_cpp && ./build.sh`
- Run the Q-learning C++ example: `./4.0_q_learning_cpp/build/qlearning`

## Validation

- There is no formal test suite, linter, or CI configuration in this repository.
- There is no single-test command today; validate by running the generator or the relevant demo executable after your change.
- For tutorial index or chapter edits, run `python merge_tutorial.py` and inspect the regenerated `RL_Tutorial_Full.md`.
- For C++ demo changes, rebuild the affected subproject and run its executable.
- Both C++ `build.sh` scripts delete and recreate their local `build/` directory before invoking CMake.

## Architecture overview

- This repository is documentation-first. `RL_Tutorial.md` is the tutorial index and reading map, while `chapters/*.md` contains the actual chapter source.
- `merge_tutorial.py` is the glue between the index and the split chapters. It parses the chapter table in `RL_Tutorial.md`, preserves the navigation and learning-path sections, and concatenates the chapter files into `RL_Tutorial_Full.md`.
- `RL_Tutorial_Full.md` is a generated artifact. Edit `RL_Tutorial.md` or files under `chapters/` instead of hand-editing the merged file.
- `3.0_bellman_cpp_cheese_example/` is a small Bellman-value teaching demo. `dag.h` defines graph nodes plus traversal/path helpers, and `bellman_update.cc` adds Bellman backup logic and greedy policy extraction on top of that DAG.
- `4.0_q_learning_cpp/` is a separate Q-learning teaching scaffold. `q_learning.h` defines actions, cell types, and the Q-table shape; `q_learning.cc` implements table initialization and cell reward updates; `main.cc` currently serves as a small smoke-test-style entrypoint.
- The numbered markdown chapters and numbered C++ demo directories are conceptually aligned as a learning sequence, so explanation changes should stay consistent across the tutorial text and the corresponding demos.

## Working assumptions in this repo

- The chapter table format in `RL_Tutorial.md` is part of the merge contract. `merge_tutorial.py` expects rows like `| 第X章 | 标题 | [file](path) |`, so changes to that table structure require updating the parser as well.
- Chapter filenames and links include Chinese characters and punctuation; keep paths exact when editing the index.
- Generated artifacts are limited and should usually be treated as derived output: `RL_Tutorial_Full.md` and each C++ subproject's `build/` directory.
- Run commands from the repository root unless a subproject build script explicitly expects to be run from inside its own directory.
