#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define LOG_INFO std::cout << __FILE__ << ":" << __LINE__ << ":"
#define LOG_ERROR std::cerr << __FILE__ << ":" << __LINE__ << ":"

// number of action space
const int kActionSpace = 4;

// actions
namespace action {
const int kActionUp = 0;
const int kActionDown = 1;
const int kActionLeft = 2;
const int kActionRight = 3;
} // namespace action

// one table that agent can move
const int kRows = 64;
const int kCols = 128;

// cell type
enum class CellType { TrapCell, NormalCell, BingoCell };
namespace cell_reward {
const double TrapCellReward = -1e6;
const double NormalCellReward = 0.0;
const double BingoCellReward = 1e3;
} // namespace cell_reward

// q-cell data
struct QCellData {
  // double rewards_[kActionSpace]{0.0, 0.0, 0.0, 0.0}; // rewards of each
  // action
  double reward_{0.0}; // reward at this state
  double qualitys_[kActionSpace]{0.0, 0.0, 0.0,
                                 0.0}; // qualities with each action
  CellType cell_type_{CellType::NormalCell};
};

using RowQCellData = std::vector<QCellData>;
using QTableData = std::vector<RowQCellData>;

// q-table
class QTable {
public:
  // init table
  void InitializeTable(int rows = kRows, int cols = kCols);

  // update cell type
  void UpdateCellType(int r, int c, CellType cell_type);

  // print whole table
  void ShowTable();

  // show cell data
  static void ShowQCellData(const QCellData &cell_data);

  // get table
  const QTableData &GetQTabelData() const { return q_table_data_; }

protected:
  QTableData q_table_data_; // q table data
  int rows_{0}, cols_{0};   // rows, cols
};

class QLearning {
public:
  QLearning(double gamma = 0.9, double alpha = 0.01)
      : gamma_(gamma), alpha_(alpha) {}

public:
  // optimize q-table
  void Optimize();

public:
  void Pi();

protected:
  double gamma_{0.9};  // gamma in bellman function, how you value future
  double alpha_{0.01}; // learning rate, how much take from TD error each step
  QTable q_table_;     // quality table
};