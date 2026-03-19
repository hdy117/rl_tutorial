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

  table.UpdateCellType(1, 2, CellType::BingoCell);
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
  q_table->UpdateCellType(23, 48, CellType::TrapCell);
  q_table->UpdateCellType(4, 51, CellType::TrapCell);
  q_table->UpdateCellType(36, 67, CellType::TrapCell);
  q_table->UpdateCellType(30, 100, CellType::TrapCell);

  q_table->UpdateCellType(kRows - 1, kCols - 1, CellType::BingoCell);
  LOG_INFO << "Update table.\n";

  // 3.0 q-learning
  double gamma = 0.9, learning_rate = 0.01;
  QLearning q_learning(gamma, learning_rate);
  q_learning.AcceptQTable(q_table);
  LOG_INFO << "build q-learning.\n";

  // 4.0 optimize
  const int kEpoches = 100;
  double eplison = 1.0;
  const int KMaxSteps = kRows * kCols;
  for (auto i = 0; i < kEpoches; i++) {
    q_learning.Optimize(eplison, KMaxSteps);
    eplison = static_cast<double>(kEpoches - i) / kEpoches;
  }
  LOG_INFO << "learning.\n";

  // 5.0 save model
  q_learning.Save("./q_learning.data");
  LOG_INFO << "saved.\n";
}

int main() {
  QLearningUnitTest();

  // 1.0 optimize
  OptimizeQLearning();

  // 2.0 load
  QLearning q_learning;
  q_learning.Load();
  LOG_INFO << "loaded.\n";

  // 3.0 Pi
  int r = 0, c = 0;
  q_learning.RandomRowCol(r, c);
  q_learning.Pi(r, c);
  LOG_INFO << "done.\n";

  return 0;
}
