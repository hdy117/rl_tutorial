#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define LOG_INFO std::cout << __FILE__ << ":" << __LINE__ << ":"
#define LOG_ERROR std::cerr << __FILE__ << ":" << __LINE__ << ":"

// number of action space
const int kActionSpace = 5;

// actions
namespace action {
const int kActionUp = 0;
const int kActionDown = 1;
const int kActionLeft = 2;
const int kActionRight = 3;
const int kActionNoMove = 4;
} // namespace action

// one table that agent can move
const int kRows = 64;
const int kCols = 128;

// cell type
enum class CellType { TrapCell, NormalCell, BingoCell };
namespace cell_reward {
const double TrapCellReward = -1e2;
const double NormalCellReward = 0.0;
const double BingoCellReward = 1e2;
} // namespace cell_reward

// initial quality of q(s,a)
const double kInitialQuality = -0.1;

// q-cell data
struct QCellData {
  double reward_{0.0};             // reward at this state
  double qualities_[kActionSpace]; // qualities with each action
  CellType cell_type_{CellType::NormalCell};

  // constructor
  QCellData() {
    for (auto &quality : qualities_) {
      quality = kInitialQuality;
    }
  }
};

// q table data
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
  // optimize q-table with eplison-greedy
  void Optimize(int n_epoches = 3000, double initial_epsilon = 1.0);

public:
  void Pi();

protected:
  double gamma_{0.9};  // gamma in bellman function, how you value future
  double alpha_{0.01}; // learning rate, how much take from TD error each step
  QTable q_table_;     // quality table
};