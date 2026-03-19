#include "q_learning.h"

void QTable::InitializeTable(int rows, int cols) {
  // update shape
  rows_ = rows;
  cols_ = cols;

  // q table
  q_table_data_.reserve(rows);

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