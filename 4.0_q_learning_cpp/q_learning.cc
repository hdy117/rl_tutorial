#include "q_learning.h"

#include <google/protobuf/util/json_util.h>

using google::protobuf::util::JsonStringToMessage;
using google::protobuf::util::MessageToJsonString;

void QTable::InitializeTable(int rows, int cols) {
  // update shape
  rows_ = rows;
  cols_ = cols;

  // clear and initialize q table data
  q_table_data_.Clear();

  // initial table
  for (auto r = 0; r < rows; r++) {
    // one row
    auto *one_row = q_table_data_.add_rows();
    for (auto c = 0; c < cols; c++) {
      auto *cell = one_row->add_cells();
      cell->set_reward(0.0);
      cell->set_cell_type(qlearning::NORMAL_CELL);
      for (int i = 0; i < kActionSpace; ++i) {
        cell->add_qualities(kInitialQuality);
      }
    }
  }

  // set default state-action quality for boarder cells
  for (auto r = 0; r < rows_; ++r) {
    auto &cell = MutableCellData(r, 0);
    cell.set_qualities(action::kActionLeft, kBoardQuality);

    auto &cell_right = MutableCellData(r, cols_ - 1);
    cell_right.set_qualities(action::kActionRight, kBoardQuality);
  }
  for (auto c = 0; c < cols_; ++c) {
    auto &cell = MutableCellData(0, c);
    cell.set_qualities(action::kActionUp, kBoardQuality);

    auto &cell_right = MutableCellData(rows_ - 1, c);
    cell_right.set_qualities(action::kActionDown, kBoardQuality);
  }
}

void QTable::ShowTable() {
  for (int r = 0; r < q_table_data_.rows_size(); ++r) {
    const auto &row = q_table_data_.rows(r);
    for (int c = 0; c < row.cells_size(); ++c) {
      ShowQCellData(row.cells(c));
      std::cout << "\t";
    }
    std::cout << "\n";
  }
}

void QTable::ShowQCellData(const QCellData &cell_data) {
  LOG_INFO << cell_data.DebugString() << "\n";
}

// update cell type
void QTable::UpdateCellType(int r, int c, CellType cell_type) {
  // early check
  if (r >= rows_ || c >= cols_) {
    LOG_ERROR << "r or c is too big, " << r << ", " << c;
    return;
  }

  // get cell
  auto &cell = MutableCellData(r, c);
  cell.set_cell_type(cell_type);

  // update reward data
  double reward = cell_reward::NormalCellReward;
  if (cell_type == qlearning::BINGO_CELL) {
    reward = cell_reward::BingoCellReward;
  } else if (cell_type == qlearning::TRAP_CELL) {
    reward = cell_reward::TrapCellReward;
  }
  cell.set_reward(reward);
}

// get max quality of cell state-actions
double QTable::MaxQualityOf(const QCellData &cell) {
  // best quality if among this state-actions
  double best_quality = -1e9;
  for (auto action_i = 0; action_i < kActionSpace; action_i++) {
    if (cell.qualities(action_i) > best_quality) {
      best_quality = cell.qualities(action_i);
    }
  }
  return best_quality;
}

// get action of max quality of cell state-actions
int QTable::ActionOfMaxQualityOf(const QCellData &cell) {
  // best quality if among this state-actions
  double best_quality = -1e9;
  int best_quality_action = action::kActionNoMove;
  for (auto action_i = 0; action_i < kActionSpace; action_i++) {
    if (cell.qualities(action_i) > best_quality) {
      best_quality = cell.qualities(action_i);
      best_quality_action = action_i;
    }
  }
  return best_quality_action;
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
int QLearning::RandomAction(int action_space) {
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
  LOG_INFO << "===============================================\n";
  LOG_INFO << "optimize start r:" << cur_r << ", c:" << cur_c << "\n";

  for (auto i = 0; i < max_steps; ++i) {
    // get current cell
    LOG_INFO << "current cell r:" << cur_r << ", c:" << cur_c << "\n";
    auto &cur_cell = q_table_->MutableCellData(cur_r, cur_c);

    // early return
    if (cur_cell.cell_type() == qlearning::BINGO_CELL) {
      LOG_INFO << "bingo cell reached, return now";
      return;
    }

    // update quality of actions at this state
    int chosen_action = action::kActionNoMove;

    // random eplison
    auto cur_epsilon = Random01();
    LOG_INFO << "cur_epsilon:" << cur_epsilon << ", eplison:" << epsilon
             << "\n";
    int next_r = cur_r, next_c = cur_c;

    // make an action with epslison used
    if (cur_epsilon < epsilon) {
      // random action, exploration
      chosen_action = RandomAction(kActionSpace);
    } else {
      // select best action based on current q_s_a
      chosen_action = q_table_->ActionOfMaxQualityOf(cur_cell);
    }

    // update chosen next cell
    int r = cur_r, c = cur_c;
    auto ret = q_table_->UpdateCellWithAction(r, c, chosen_action);
    double max_next_q_s_a = kBoardQuality;
    double rewart_t_plus_1 = cell_reward::TrapCellReward;
    if (ret) {
      // action with random action
      next_r = r;
      next_c = c;

      const auto &next_cell = q_table_->MutableCellData(next_r, next_c);
      max_next_q_s_a = q_table_->MaxQualityOf(next_cell);
      rewart_t_plus_1 = next_cell.reward();

      LOG_INFO << "next cell r:" << next_r << ", c:" << next_c << "\n";
      LOG_INFO << "chosen_action:" << chosen_action
               << ", max_q_s_a:" << cur_cell.qualities(chosen_action) << "\n";
    } else {
      // penaulty to cross boarder
      max_next_q_s_a = kBoardQuality;
      rewart_t_plus_1 = cell_reward::TrapCellReward;
    }

    // update quality of this state-action, reward + arg max(Q(s',a')) vs a'
    double bellman_target = rewart_t_plus_1 + gamma_ * max_next_q_s_a;
    double q_s_a = cur_cell.qualities(chosen_action); //  Q(s,a)
    double td_error = bellman_target - q_s_a;
    cur_cell.set_qualities(chosen_action, q_s_a + alpha_ * td_error);

    // update r,c
    cur_r = next_r;
    cur_c = next_c;
  }
}

// save q-learning using proto JSON
void QLearning::Save(const std::string &data_file) {
  qlearning::QLearningModel model;
  model.set_rows(q_table_->rows_);
  model.set_cols(q_table_->cols_);
  *model.mutable_q_table_data() = q_table_->GetQTabelData();

  std::string json_string;
  auto status = MessageToJsonString(model, &json_string);
  if (!status.ok()) {
    LOG_ERROR << "failed to serialize model to JSON: " << status.ToString()
              << "\n";
    return;
  }

  std::ofstream ofs(data_file);
  if (!ofs.is_open()) {
    LOG_ERROR << "failed to open file for saving: " << data_file << "\n";
    return;
  }

  ofs << json_string;
  ofs.close();
  LOG_INFO << "q-learning model saved to: " << data_file << "\n";
}

// load q-learning using proto JSON
void QLearning::Load(const std::string &data_file) {
  std::ifstream ifs(data_file);
  if (!ifs.is_open()) {
    LOG_ERROR << "failed to open file for loading: " << data_file << "\n";
    return;
  }

  std::string json_string((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
  ifs.close();

  qlearning::QLearningModel model;
  auto status = JsonStringToMessage(json_string, &model);
  if (!status.ok()) {
    LOG_ERROR << "failed to parse JSON: " << status.ToString() << "\n";
    return;
  }

  // create q_table_ if not exists
  if (!q_table_) {
    q_table_ = std::make_shared<QTable>();
  }

  // reinitialize table if size changed
  if (model.rows() != q_table_->rows_ || model.cols() != q_table_->cols_) {
    q_table_->InitializeTable(model.rows(), model.cols());
  }

  // copy q-table data
  q_table_->MutableQTableData() = model.q_table_data();

  LOG_INFO << "q-learning model loaded from: " << data_file << "\n";
}

// find path to bingo point
void QLearning::Pi(int r, int c, int max_steps) {
  // current cell state

  bool bingo = false;
  int cur_r = r, cur_c = c, step_counter = 0;

  while (!bingo && step_counter < max_steps) {

    int chosen_action = action::kActionNoMove;
    double best_q_s_a = kBoardQuality;

    // get current state
    const auto &cur_cell = q_table_->MutableCellData(cur_r, cur_c);

    // print cell info
    LOG_INFO << "===============step:" << step_counter << "===============\n";
    LOG_INFO << "cur_r:" << cur_r << ", cur_c:" << cur_c << "\n";
    q_table_->ShowQCellData(cur_cell);

    // check if find bingo
    if (cur_cell.cell_type() == qlearning::BINGO_CELL) {
      LOG_INFO << "bingo!!!\n";
      return;
    }

    // find best action in this state
    for (auto action_i = 0; action_i < kActionSpace; ++action_i) {
      if (cur_cell.qualities(action_i) > best_q_s_a) {
        chosen_action = action_i;
        best_q_s_a = cur_cell.qualities(action_i);
      }
    }

    LOG_INFO << "best action is " << chosen_action
             << ", best_q_s_a:" << best_q_s_a << "\n";

    // act
    int r = cur_r, c = cur_c;
    if (q_table_->UpdateCellWithAction(r, c, chosen_action)) {
      cur_r = r;
      cur_c = c;
    }

    // update step counter
    step_counter++;
  }
}
