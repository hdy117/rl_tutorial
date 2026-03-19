#include "q_learning.h"
#include <fstream>
#include <iostream>

void PrintSeperator() { LOG_INFO << "===============================\n"; }

// unit test
void QLearningUnitTest() {
  PrintSeperator();

  QTable table;
  table.InitializeTable(2, 4);

  table.ShowTable();

  PrintSeperator();

  table.UpdateCellType(1, 2, qlearning::BINGO_CELL);
  table.ShowTable();

  PrintSeperator();
}

// check if file exists
bool FileExists(const std::string &filename) {
  std::ifstream file(filename);
  return file.good();
}

// optimize q-learning with resume support
void OptimizeQLearning(bool resume = false) {
  const std::string kModelPath = "./q_learning.json";

  // 1.0 prepare q-learning
  PrintSeperator();
  double gamma = 0.9, learning_rate = 0.1;
  QLearning q_learning(gamma, learning_rate);

  if (resume && FileExists(kModelPath)) {
    // Load existing model
    q_learning.Load(kModelPath);
    LOG_INFO << "Resumed from existing model: " << kModelPath << "\n";
  } else {
    // Build new q table
    QTablePtr q_table = std::make_shared<QTable>();
    q_table->InitializeTable(kRows, kCols);

    // Set trap and bingo cells
    q_table->UpdateCellType(23, 48, qlearning::TRAP_CELL);
    q_table->UpdateCellType(4, 51, qlearning::TRAP_CELL);
    q_table->UpdateCellType(36, 67, qlearning::TRAP_CELL);
    q_table->UpdateCellType(30, 100, qlearning::TRAP_CELL);
    q_table->UpdateCellType(kRows - 1, kCols - 1, qlearning::BINGO_CELL);

    q_learning.AcceptQTable(q_table);
    LOG_INFO << "Initialized new Q-table.\n";
  }

  // 2.0 optimize
  PrintSeperator();
  const int kEpoches = resume ? 5000 : 20000; // Fewer epochs if resuming
  double epsilon = resume ? 0.3 : 1.0;        // Lower epsilon if resuming
  const int KMaxSteps = kRows * kCols;

  LOG_INFO << "Training for " << kEpoches
           << " epochs, starting epsilon=" << epsilon << "\n";

  for (auto i = 0; i < kEpoches; i++) {
    q_learning.Optimize(epsilon, KMaxSteps);
    epsilon = static_cast<double>(kEpoches - i) / kEpoches;
  }
  LOG_INFO << "Training completed.\n";

  // 3.0 save model
  PrintSeperator();
  q_learning.Save(kModelPath);
  LOG_INFO << "Model saved to " << kModelPath << "\n";
}

int main(int argc, char *argv[]) {
  // QLearningUnitTest();
  int mode = 1; // 1=train, 0=play, 2=force fresh training

  if (argc >= 2) {
    mode = std::stoi(argv[1]);
  }
  LOG_INFO << "mode:" << mode << "\n";

  const std::string kModelPath = "./q_learning.json";

  if (mode == 1) {
    // Training mode - auto resume if model exists
    bool should_resume = FileExists(kModelPath);
    if (should_resume) {
      LOG_INFO << "Existing model found, resuming training.\n";
    }
    OptimizeQLearning(/*resume=*/should_resume);
  } else if (mode == 2) {
    // Force fresh training (ignore existing model)
    LOG_INFO << "Force fresh training.\n";
    OptimizeQLearning(/*resume=*/false);
  } else {
    // Play mode
    PrintSeperator();
    QLearning q_learning;
    if (!FileExists(kModelPath)) {
      LOG_ERROR << "No trained model found. Please train first (mode=1).\n";
      return 1;
    }
    q_learning.Load(kModelPath);
    LOG_INFO << "Model loaded.\n";

    // Pi
    PrintSeperator();
    int r = 0, c = 0;
    q_learning.RandomRowCol(r, c);
    q_learning.Pi(r, c, 100);
    LOG_INFO << "done.\n";
  }

  return 0;
}
