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

// q-learning demo
void QLearningDemo() { PrintSeperator(); }

int main() {
  QLearningUnitTest();
  return 0;
}