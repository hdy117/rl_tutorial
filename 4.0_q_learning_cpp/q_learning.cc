#include "q_learning.h"

void QTable::InitializeTable(int rows, int cols) {
  // update shape
  rows_ = rows;
  cols_ = cols;

  // q table
  q_table_data_.reserve(rows);

  // initial table
  for (auto r = 0; r < rows; r++) {
    // one row
    RowQCellData one_row;
    one_row.reserve(cols);
    for (auto c = 0; c < cols; c++) {
      one_row.push_back(QCellData());
    }

    // save this row
    q_table_data_.push_back(one_row);
  }

  // set default state-action quality for boarder cells
  for (auto r = 0; r < rows_; ++r) {
    auto &cell = q_table_data_.at(r).at(0);
    cell.qualities_[action::kActionLeft] = kBoardQuality;

    auto &cell_right = q_table_data_.at(r).at(cols_ - 1);
    cell_right.qualities_[action::kActionRight] = kBoardQuality;
  }
  for (auto c = 0; c < cols_; ++c) {
    auto &cell = q_table_data_.at(0).at(c);
    cell.qualities_[action::kActionUp] = kBoardQuality;

    auto &cell_right = q_table_data_.at(rows_ - 1).at(c);
    cell_right.qualities_[action::kActionDown] = kBoardQuality;
  }
}

void QTable::ShowTable() {
  for (const auto &row : q_table_data_) {
    for (const auto &cell : row) {
      ShowQCellData(cell);
      std::cout << "\t";
    }
    std::cout << "\n";
  }
}

void QTable::ShowQCellData(const QCellData &cell_data) {
  std::cout << "[Reward: ";
  std::cout << cell_data.reward_;

  std::cout << "] [Q-Values: ";
  for (int i = 0; i < kActionSpace; ++i) {
    std::cout << cell_data.qualities_[i];
    if (i < kActionSpace - 1)
      std::cout << ", ";
  }
  std::cout << "]";
}

// update cell type
void QTable::UpdateCellType(int r, int c, CellType cell_type) {
  // early check
  if (r >= rows_ || c >= cols_) {
    LOG_ERROR << "r or c is too big, " << r << ", " << c;
    return;
  }

  // get cell
  auto &cell = q_table_data_.at(r).at(c);
  cell.cell_type_ = cell_type;

  // update reward data
  cell.reward_ = cell_reward::NormalCellReward;
  if (cell.cell_type_ == CellType::BingoCell) {
    cell.reward_ = cell_reward::BingoCellReward;
  } else if (cell.cell_type_ == CellType::TrapCell) {
    cell.reward_ = cell_reward::TrapCellReward;
  }
}

// get max quality of cell state-actions
double QTable::MaxQualityOf(const QCellData &cell) {
  // best quality if among this state-actions
  double best_quality = -1e9;
  for (auto action_i = 0; action_i < kActionSpace; action_i++) {
    if (cell.qualities_[action_i] > best_quality) {
      best_quality = cell.qualities_[action_i];
    }
  }
  return best_quality;
}

// random r, c
void QLearning::RandomRowCol(int &r, int &c) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> row_dist(0, q_table_->rows_ - 1);
  std::uniform_int_distribution<> col_dist(0, q_table_->cols_ - 1);

  r = row_dist(gen);
  c = col_dist(gen);
}

// random double, (0,1.0)
double QLearning::Random01() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  return dis(gen);
}

// random action
int RandomAction(int action_space) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, action_space - 1);
  return dis(gen);
}

// optimize q-table with eplison-greedy, also limit agent max steps in case of
// stuck somewhere
void QLearning::Optimize(double epsilon, int max_steps) {
  // initial position
  int cur_r = 0, cur_c = 0;

  // random start position
  RandomRowCol(cur_r, cur_c);
  LOG_INFO << "optimize start r:" << cur_r << ", c:" << cur_c << "\n";

  for (auto i = 0; i < max_steps; ++i) {
    // get current cell
    auto &cur_cell = q_table_->MutableCellData(cur_r, cur_c);

    // update quality of actions at this state
    for (auto action_i = 0; action_i < kActionSpace; ++action_i) {
      // next state with current state-action
      int r = cur_r, c = cur_c;
      double max_next_q_s_a = kBoardQuality;
      auto ret = q_table_->UpdateCellWithAction(r, c, action_i);
      if (ret) {
        auto &next_cell = q_table_->MutableCellData(r, c);
        max_next_q_s_a = q_table_->MaxQualityOf(next_cell);
      }

      // update quality of this state-action, reward + arg max(Q(s',a')) vs a'
      double bellman_target = cur_cell.reward_ + gamma_ * max_next_q_s_a;
      double q_s_a = cur_cell.qualities_[action_i]; //  Q(s,a)
      double td_error = bellman_target - q_s_a;
      cur_cell.qualities_[action_i] = q_s_a + alpha_ * td_error;
    }

    // make an action with epslison used
    auto cur_eplison = Random01();
    LOG_INFO << "cur random eplison:" << cur_eplison
             << ", input eplison:" << epsilon << "\n";
    if (cur_eplison < epsilon) {
      // random action, exploration
      auto random_action_i = RandomAction(kActionSpace);
      int r = cur_r, c = cur_c;
      auto ret = q_table_->UpdateCellWithAction(r, c, random_action_i);
      if (ret) {
        // action with random action
        cur_r = r;
        cur_c = c;
      }
    } else {
      // fix action, exploitation
      int best_action = action::kActionNoMove;
      double best_q_s_a = kBoardQuality;
      for (auto action_i = 0; action_i < kActionSpace; ++action_i) {
        if (cur_cell.qualities_[action_i] > best_q_s_a) {
          best_q_s_a = cur_cell.qualities_[action_i];
          best_action = action_i;
        }
      }
      // action among current best q_s_a
      q_table_->UpdateCellWithAction(cur_r, cur_c, best_action);
    }
  }
}

// find path to bingo point
void QLearning::Pi(int r, int c) {
  // current cell state

  bool bingo = false;
  int cur_r = r, cur_c = c;

  while (!bingo) {

    int best_action = action::kActionNoMove;
    double best_q_s_a = kBoardQuality;

    // get current state
    const auto &cur_cell = q_table_->MutableCellData(cur_r, cur_c);

    // print cell info
    LOG_INFO << "==============================\n";
    q_table_->ShowQCellData(cur_cell);

    // check if find bingo
    if (cur_cell.cell_type_ == CellType::BingoCell) {
      LOG_INFO << "bingo!!!\n";
      return;
    }

    // find best action in this state
    for (auto action_i = 0; action_i < kActionSpace; ++action_i) {
      if (cur_cell.qualities_[action_i] > best_q_s_a) {
        best_action = action_i;
        best_q_s_a = cur_cell.qualities_[action_i];
      }
    }

    // act
    q_table_->UpdateCellWithAction(cur_r, cur_c, best_action);
  }
}