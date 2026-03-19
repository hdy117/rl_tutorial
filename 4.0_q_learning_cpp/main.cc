#include "q_learning.h"
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

// optimize q-learning
void OptimizeQLearning() {
  // 1.0 build q table
  PrintSeperator();
  QTablePtr q_table = std::make_shared<QTable>();
  q_table->InitializeTable(kRows, kCols);
  LOG_INFO << "InitializeTable.\n";

  // 2.0 set bingo and trap cell
  PrintSeperator();
  q_table->UpdateCellType(23, 48, qlearning::TRAP_CELL);
  q_table->UpdateCellType(4, 51, qlearning::TRAP_CELL);
  q_table->UpdateCellType(36, 67, qlearning::TRAP_CELL);
  q_table->UpdateCellType(30, 100, qlearning::TRAP_CELL);

  q_table->UpdateCellType(kRows - 1, kCols - 1, qlearning::BINGO_CELL);
  LOG_INFO << "Update table.\n";

  // 3.0 q-learning
  PrintSeperator();
  double gamma = 0.9, learning_rate = 0.01;
  QLearning q_learning(gamma, learning_rate);
  q_learning.AcceptQTable(q_table);
  LOG_INFO << "build q-learning.\n";

  // 4.0 optimize
  PrintSeperator();
  const int kEpoches = 1000;
  double eplison = 1.0;
  const int KMaxSteps = kRows * kCols;
  for (auto i = 0; i < kEpoches; i++) {
    q_learning.Optimize(eplison, KMaxSteps);
    eplison = static_cast<double>(kEpoches - i) / kEpoches;
  }
  LOG_INFO << "learning.\n";

  // 5.0 save model
  PrintSeperator();
  q_learning.Save("./q_learning.json");
  LOG_INFO << "saved.\n";
}

int main(int argc, char *argv[]) {
  // QLearningUnitTest();
  int if_optimize = 0;

  if (argc == 2) {
    if_optimize = std::stoi(argv[1]);
  }
  LOG_INFO << "if_optimize:" << if_optimize;

  if (if_optimize) {
    // 1.0 optimize
    PrintSeperator();
    OptimizeQLearning();
  }

  // 2.0 load
  PrintSeperator();
  QLearning q_learning;
  q_learning.Load();
  LOG_INFO << "loaded.\n";

  // 3.0 Pi
  PrintSeperator();
  int r = 0, c = 0;
  q_learning.RandomRowCol(r, c);
  q_learning.Pi(r, c, 100);
  LOG_INFO << "done.\n";

  return 0;
}
