# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

- Install Python dependencies used by the tutorial tooling: `pip install -r requirements.txt`
- Regenerate the merged tutorial document from the repo root: `python merge_tutorial.py`
- Build the Bellman DAG example: `cd 3.0_bellman_cpp_cheese_example && ./build.sh`
- Run the Bellman DAG example: `cd 3.0_bellman_cpp_cheese_example && ./build/bellman_update_cheese_example`
- Build the Q-learning C++ example: `cd 4.0_q_learning_cpp && ./build.sh`
- Train or resume the Q-learning model: `cd 4.0_q_learning_cpp && ./build/qlearning 1`
- Force a fresh Q-learning training run: `cd 4.0_q_learning_cpp && ./build/qlearning 2`
- Run greedy policy playback from a saved model: `cd 4.0_q_learning_cpp && ./build/qlearning 0`
- Build the Qt Q-table visualizer: `cd 4.0_q_learning_cpp && ./build_qt.sh`
- Run the Qt visualizer against the saved model: `cd 4.0_q_learning_cpp/build_qt && ./qt_visualizer ../q_learning.json`

## Validation

- There is no formal test suite, linter, or CI configuration in this repository.
- There is no single-test command today; validate by rerunning the relevant generator or demo executable after your change.
- For tutorial index or chapter edits, run `python merge_tutorial.py` and inspect the regenerated `RL_Tutorial_Full.md`.
- For Bellman DAG demo changes, rebuild and run `cd 3.0_bellman_cpp_cheese_example && ./build/bellman_update_cheese_example`.
- For Q-learning core changes, rebuild and run the mode that exercises the code path you changed: `1` for train/resume, `2` for fresh training, `0` for playback.
- For visualizer changes, rerun `cd 4.0_q_learning_cpp && ./build_qt.sh` and start `./build_qt/qt_visualizer` with a known-good `q_learning.json`.
- Both `3.0_bellman_cpp_cheese_example/build.sh` and `4.0_q_learning_cpp/build.sh` delete and recreate their local `build/` directory before invoking CMake.

## Architecture overview

- This repository is documentation-first. `RL_Tutorial.md` is the tutorial index and reading map, `chapters/*.md` contains the actual chapter source, and `RL_Tutorial_Full.md` is a generated artifact.
- `merge_tutorial.py` is the glue between the index and the split chapters. It parses chapter rows from the markdown table in `RL_Tutorial.md`, preserves the global navigation diagram plus the learning-path section, and concatenates chapter files in chapter-number order.
- `3.0_bellman_cpp_cheese_example/` is a Bellman-value teaching demo. `dag.h` defines graph nodes plus traversal and path-finding helpers; `bellman_update.cc` layers Bellman backup logic and greedy policy extraction on top of that DAG, including both an intentionally incomplete updater and the corrected recursive max-backup version.
- `4.0_q_learning_cpp/` is the gridworld Q-learning teaching scaffold. `q_learning.proto` defines the persisted model schema; `q_learning.h` and `q_learning.cc` implement the Q-table, action semantics, Bellman-style update logic, and save/load behavior; `main.cc` is the CLI entrypoint for train/resume/play modes.
- The Q-learning example persists its learned table as protobuf JSON in `q_learning.json`. That file is both training output and playback input.
- `4.0_q_learning_cpp/qt_visualizer.cc` is a separate Qt6 desktop inspector for `q_learning.json`. It is built through `build_qt.sh` and `CMakeLists_qt.txt`, not through the main `4.0_q_learning_cpp/CMakeLists.txt`.
- The numbered markdown chapters and numbered C++ demo directories are conceptually aligned as a learning sequence, so explanation changes should stay consistent across the tutorial text and the corresponding demos.

## Working assumptions in this repo

- The chapter table format in `RL_Tutorial.md` is part of the merge contract. `merge_tutorial.py` expects rows like `| 第X章 | 标题 | [file](path) |`, so changes to that table structure require updating the parser as well.
- Chapter filenames and links include Chinese characters and punctuation; keep paths exact when editing the index.
- `q_learning.json` is resolved relative to the process working directory, so the Q-learning commands are safest when run from inside `4.0_q_learning_cpp/`.
- Treat `RL_Tutorial_Full.md`, each C++ subproject's `build/` directory, `4.0_q_learning_cpp/build_qt/`, and protobuf files generated under build directories as derived artifacts; edit the checked-in sources instead.
- Do not hand-edit the copied files inside `4.0_q_learning_cpp/build_qt/`; change `4.0_q_learning_cpp/qt_visualizer.cc` or `4.0_q_learning_cpp/CMakeLists_qt.txt` and rerun `./build_qt.sh`.
- Run commands from the repository root unless a subproject build or run command explicitly expects to be run from inside its own directory, as the Q-learning commands do.
