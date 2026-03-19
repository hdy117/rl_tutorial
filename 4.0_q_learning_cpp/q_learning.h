#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "q_learning.pb.h"

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

// cell type (alias for proto enum)
using CellType = qlearning::CellType;
namespace cell_reward {
const double TrapCellReward = -1e2;
const double NormalCellReward = 0.0;
const double BingoCellReward = 1e2;
} // namespace cell_reward

// initial quality of q(s,a)， negative quality will speed up find shortest path
// although gamma will help to find shortest path
const double kInitialQuality = 0.0;
const double kBoardQuality = -1e3;

// alias for proto-generated types
using QCellData = qlearning::QCellData;
using RowQCellData = qlearning::RowQCellData;
using QTableData = qlearning::QTableData;

class QTable;
using QTablePtr = std::shared_ptr<QTable>;

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

  // mutable table data (for save/load)
  QTableData &MutableQTableData() { return q_table_data_; }

  // mutable cell data
  QCellData &MutableCellData(int r, int c) {
    return *q_table_data_.mutable_rows(r)->mutable_cells(c);
  }

  // is valid action
  bool UpdateCellWithAction(int &cur_r, int &cur_c, int action) {
    int &r = cur_r, &c = cur_c;
    if (action == action::kActionDown) {
      r++;
    } else if (action == action::kActionUp) {
      r--;
    } else if (action == action::kActionLeft) {
      c--;
    } else if (action == action::kActionRight) {
      c++;
    }

    if (r >= 0 && r < rows_ && c >= 0 && c < cols_) {
      return true;
    }

    return false;
  }

  // get max quality of cell state-actions
  double MaxQualityOf(const QCellData &cell);

public:
  int rows_{0}, cols_{0}; // rows, cols

public:
  QTableData q_table_data_; // q table data
};

class QLearning {
public:
  QLearning(double gamma = 0.9, double alpha = 0.01)
      : gamma_(gamma), alpha_(alpha) {}

public:
  // set bingo and trap cell
  void AcceptQTable(QTablePtr q_table) { q_table_ = q_table; }

  // optimize q-table with eplison-greedy, also limit agent max steps in case of
  // stuck somewhere
  void Optimize(double epsilon = 1.0, int max_steps = kRows * kCols);

public:
  // find path to bingo point
  void Pi(int r, int c, int max_steps = kRows * kCols);

public:
  // random double, (0,1.0)
  double Random01();

  // random r, c
  void RandomRowCol(int &r, int &c);

  // random action
  int RandomAction(int action_space = kActionSpace);

  // save q-learning
  void Save(const std::string &data_file = "./q_learning.json");

  // load q-learning
  void Load(const std::string &data_file = "./q_learning.json");

protected:
  double gamma_{0.9};  // gamma in bellman function, how you value future
  double alpha_{0.01}; // learning rate, how much take from TD error each step
  QTablePtr q_table_;  // quality table
};
