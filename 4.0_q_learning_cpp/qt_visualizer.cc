#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QFile>
#include <QMessageBox>
#include <QStatusBar>
#include <QPushButton>
#include <QShortcut>
#include <QKeySequence>
#include <QFrame>
#include <QGridLayout>
#include <QResizeEvent>
#include <cmath>

#include <google/protobuf/util/json_util.h>
#include "q_learning.pb.h"

using google::protobuf::util::JsonStringToMessage;

// Action indices (matches q_learning.h)
const int ACTION_UP = 0;
const int ACTION_DOWN = 1;
const int ACTION_LEFT = 2;
const int ACTION_RIGHT = 3;

// Minimum cell size to keep readable
const int MIN_CELL_SIZE = 8;

// Cell data structure
struct CellData {
  double reward = 0.0;
  std::vector<double> qualities; // [up, down, left, right, nomove]
  qlearning::CellType cell_type = qlearning::NORMAL_CELL;
  
  double upQuality() const { return qualities.size() > ACTION_UP ? qualities[ACTION_UP] : 0.0; }
  double downQuality() const { return qualities.size() > ACTION_DOWN ? qualities[ACTION_DOWN] : 0.0; }
  double leftQuality() const { return qualities.size() > ACTION_LEFT ? qualities[ACTION_LEFT] : 0.0; }
  double rightQuality() const { return qualities.size() > ACTION_RIGHT ? qualities[ACTION_RIGHT] : 0.0; }
  double noMoveQuality() const { return qualities.size() > 4 ? qualities[4] : 0.0; }
};

// Q-Learning data loader using Protobuf
class QLearningData {
public:
  int rows = 0;
  int cols = 0;
  std::vector<std::vector<CellData>> grid;
  
  bool load(const QString& filepath) {
    QFile file(filepath);
    if (!file.open(QIODevice::ReadOnly)) {
      return false;
    }
    
    QByteArray data = file.readAll();
    file.close();
    
    // Use protobuf to parse JSON
    qlearning::QLearningModel model;
    auto status = JsonStringToMessage(data.toStdString(), &model);
    if (!status.ok()) {
      return false;
    }
    
    rows = model.rows();
    cols = model.cols();
    
    grid.resize(rows, std::vector<CellData>(cols));
    
    const auto& qTableData = model.q_table_data();
    
    for (int r = 0; r < rows && r < qTableData.rows_size(); ++r) {
      const auto& rowData = qTableData.rows(r);
      for (int c = 0; c < cols && c < rowData.cells_size(); ++c) {
        const auto& cell = rowData.cells(c);
        grid[r][c].reward = cell.reward();
        grid[r][c].cell_type = cell.cell_type();
        
        for (int i = 0; i < cell.qualities_size(); ++i) {
          grid[r][c].qualities.push_back(cell.qualities(i));
        }
      }
    }
    
    return true;
  }
  
  // Get color based on max Q value (heatmap style)
  QColor getHeatmapColor(double max_q) const {
    // Normalize Q value to 0-1 range
    // Assuming Q values roughly range from -1000 to +100
    double min_q = -500;
    double max_val = 100;
    double normalized = (max_q - min_q) / (max_val - min_q);
    normalized = std::max(0.0, std::min(1.0, normalized));
    
    // Heatmap: blue (cold/low) -> green -> yellow -> red (hot/high)
    int r, g, b;
    if (normalized < 0.25) {
      // Blue to Cyan
      r = 0;
      g = (int)(255 * normalized * 4);
      b = 255;
    } else if (normalized < 0.5) {
      // Cyan to Green
      r = 0;
      g = 255;
      b = (int)(255 * (1 - (normalized - 0.25) * 4));
    } else if (normalized < 0.75) {
      // Green to Yellow
      r = (int)(255 * (normalized - 0.5) * 4);
      g = 255;
      b = 0;
    } else {
      // Yellow to Red
      r = 255;
      g = (int)(255 * (1 - (normalized - 0.75) * 4));
      b = 0;
    }
    return QColor(r, g, b);
  }
  
  // Get color based on best action direction
  QColor getDirectionalColor(int best_action) const {
    // Directional colors: Up=Red, Down=Blue, Left=Green, Right=Yellow, NoMove=Gray
    switch (best_action) {
      case ACTION_UP:    return QColor(255, 100, 100); // Red-ish
      case ACTION_DOWN:  return QColor(100, 100, 255); // Blue-ish
      case ACTION_LEFT:  return QColor(100, 200, 100); // Green-ish
      case ACTION_RIGHT: return QColor(255, 200, 100); // Orange/Yellow-ish
      default:           return QColor(180, 180, 180); // Gray
    }
  }
  
  QColor getCellColor(int r, int c, bool use_directional = false) const {
    const auto& cell = grid[r][c];
    if (cell.cell_type == qlearning::BINGO_CELL) {
      return QColor(50, 180, 50);   // Deep green for bingo
    } else if (cell.cell_type == qlearning::TRAP_CELL) {
      return QColor(180, 50, 50);   // Deep red for trap
    }
    
    // Normal cell
    double max_q = -1e9;
    int best_action = 4; // Default to NoMove
    for (int i = 0; i < (int)cell.qualities.size() && i < 5; ++i) {
      if (cell.qualities[i] > max_q) {
        max_q = cell.qualities[i];
        best_action = i;
      }
    }
    
    if (use_directional) {
      return getDirectionalColor(best_action);
    } else {
      return getHeatmapColor(max_q);
    }
  }
  
  // Get best action index for a cell
  int getBestAction(int r, int c) const {
    const auto& cell = grid[r][c];
    double max_q = -1e9;
    int best_action = 4;
    for (int i = 0; i < (int)cell.qualities.size() && i < 5; ++i) {
      if (cell.qualities[i] > max_q) {
        max_q = cell.qualities[i];
        best_action = i;
      }
    }
    return best_action;
  }
  
  static QString cellTypeString(qlearning::CellType type) {
    switch (type) {
      case qlearning::BINGO_CELL: return "Bingo";
      case qlearning::TRAP_CELL: return "Trap";
      default: return "Normal";
    }
  }
};

// Arrow display widget
class ArrowWidget : public QWidget {
  Q_OBJECT
public:
  ArrowWidget(QWidget* parent = nullptr) : QWidget(parent), quality_(0.0), angle_(0) {
    setFixedSize(80, 80);
  }
  
  void setArrow(double quality, double angle, const QString& label) {
    quality_ = quality;
    angle_ = angle;
    label_ = label;
    update();
  }
  
protected:
  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    int cx = width() / 2;
    int cy = height() / 2;
    
    painter.save();
    painter.translate(cx, cy);
    painter.rotate(angle_);
    
    // Arrow color based on quality
    QColor color = getQualityColor(quality_);
    painter.setPen(QPen(color, 3));
    painter.setBrush(color);
    
    // Arrow shaft
    int len = 25;
    painter.drawLine(-len/2, 0, len/2, 0);
    
    // Arrow head
    QPointF points[3] = {
      QPointF(len/2, 0),
      QPointF(len/2 - 8, -6),
      QPointF(len/2 - 8, 6)
    };
    painter.drawPolygon(points, 3);
    
    painter.restore();
    
    // Draw label
    painter.setPen(Qt::black);
    painter.setFont(QFont("Arial", 9, QFont::Bold));
    QRect labelRect(0, height() - 20, width(), 20);
    painter.drawText(labelRect, Qt::AlignCenter, label_);
    
    // Draw value
    painter.setFont(QFont("Arial", 8));
    painter.setPen(color.darker(150));
    QRect valueRect(0, 5, width(), 18);
    painter.drawText(valueRect, Qt::AlignCenter, QString::number(quality_, 'f', 2));
  }
  
  QColor getQualityColor(double q) const {
    if (q > 0) {
      int g = std::min(255, (int)(100 + q * 100));
      return QColor(0, g, 0);
    } else if (q < -500) {
      return QColor(150, 0, 0);
    } else {
      int r = std::min(255, (int)(100 - q * 10));
      return QColor(r, 0, 0);
    }
  }
  
private:
  double quality_;
  double angle_;
  QString label_;
};

// Right panel for cell details
class CellInfoPanel : public QFrame {
  Q_OBJECT
public:
  CellInfoPanel(QWidget* parent = nullptr) : QFrame(parent) {
    setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    setFixedWidth(250);
    setStyleSheet("QFrame { background-color: #f8f8f8; }");
    
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(15, 15, 15, 15);
    
    // Title
    QLabel* title = new QLabel("📊 Cell Information", this);
    title->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #333; }");
    title->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(title);
    
    // Separator
    QFrame* line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setStyleSheet("QFrame { color: #ccc; }");
    mainLayout->addWidget(line);
    
    // Position info
    positionLabel_ = new QLabel("Position: -", this);
    positionLabel_->setStyleSheet("QLabel { font-size: 13px; }");
    mainLayout->addWidget(positionLabel_);
    
    // Cell type
    typeLabel_ = new QLabel("Type: -", this);
    typeLabel_->setStyleSheet("QLabel { font-size: 13px; }");
    mainLayout->addWidget(typeLabel_);
    
    // Reward
    rewardLabel_ = new QLabel("Reward: -", this);
    rewardLabel_->setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #0066cc; }");
    mainLayout->addWidget(rewardLabel_);
    
    // Separator
    QFrame* line2 = new QFrame(this);
    line2->setFrameShape(QFrame::HLine);
    line2->setStyleSheet("QFrame { color: #ccc; }");
    mainLayout->addWidget(line2);
    
    // Q-Values title
    QLabel* qvaluesTitle = new QLabel("🎯 Q-Values (Action Quality)", this);
    qvaluesTitle->setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #333; }");
    mainLayout->addWidget(qvaluesTitle);
    
    // Arrows grid
    QGridLayout* arrowLayout = new QGridLayout();
    arrowLayout->setSpacing(10);
    
    // Up arrow
    upArrow_ = new ArrowWidget(this);
    upArrow_->setArrow(0, -90, "UP");
    arrowLayout->addWidget(upArrow_, 0, 1, Qt::AlignCenter);
    
    // Left arrow
    leftArrow_ = new ArrowWidget(this);
    leftArrow_->setArrow(0, 180, "LEFT");
    arrowLayout->addWidget(leftArrow_, 1, 0, Qt::AlignCenter);
    
    // Center (No Move)
    noMoveLabel_ = new QLabel("NO MOVE\n-", this);
    noMoveLabel_->setAlignment(Qt::AlignCenter);
    noMoveLabel_->setStyleSheet("QLabel { font-size: 10px; color: #666; background-color: #e0e0e0; border-radius: 5px; padding: 5px; }");
    noMoveLabel_->setFixedSize(70, 70);
    arrowLayout->addWidget(noMoveLabel_, 1, 1, Qt::AlignCenter);
    
    // Right arrow
    rightArrow_ = new ArrowWidget(this);
    rightArrow_->setArrow(0, 0, "RIGHT");
    arrowLayout->addWidget(rightArrow_, 1, 2, Qt::AlignCenter);
    
    // Down arrow
    downArrow_ = new ArrowWidget(this);
    downArrow_->setArrow(0, 90, "DOWN");
    arrowLayout->addWidget(downArrow_, 2, 1, Qt::AlignCenter);
    
    mainLayout->addLayout(arrowLayout);
    
    // Best action
    bestActionLabel_ = new QLabel("Best Action: -", this);
    bestActionLabel_->setStyleSheet("QLabel { font-size: 13px; font-weight: bold; color: #009900; margin-top: 10px; }");
    mainLayout->addWidget(bestActionLabel_);
    
    mainLayout->addStretch();
    
    // Instructions
    QLabel* hint = new QLabel("💡 Hover over grid cells\nto view details", this);
    hint->setStyleSheet("QLabel { font-size: 11px; color: #888; font-style: italic; }");
    hint->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(hint);
  }
  
  void updateCellInfo(int row, int col, const CellData& cell) {
    positionLabel_->setText(QString("📍 Position: [%1, %2]").arg(row).arg(col));
    typeLabel_->setText(QString("🏷️ Type: %1").arg(QLearningData::cellTypeString(cell.cell_type)));
    
    // Color code the reward
    QString rewardColor = cell.reward > 0 ? "#009900" : (cell.reward < 0 ? "#cc0000" : "#666666");
    rewardLabel_->setText(QString("💰 Reward: <span style='color:%1;'>%2</span>")
      .arg(rewardColor)
      .arg(cell.reward, 0, 'f', 2));
    
    // Update arrows
    upArrow_->setArrow(cell.upQuality(), -90, "UP");
    downArrow_->setArrow(cell.downQuality(), 90, "DOWN");
    leftArrow_->setArrow(cell.leftQuality(), 180, "LEFT");
    rightArrow_->setArrow(cell.rightQuality(), 0, "RIGHT");
    
    // Update no move
    noMoveLabel_->setText(QString("NO MOVE\n%1").arg(cell.noMoveQuality(), 0, 'f', 2));
    
    // Find best action
    double bestQ = cell.noMoveQuality();
    QString bestAction = "NO MOVE";
    
    if (cell.upQuality() > bestQ) { bestQ = cell.upQuality(); bestAction = "UP"; }
    if (cell.downQuality() > bestQ) { bestQ = cell.downQuality(); bestAction = "DOWN"; }
    if (cell.leftQuality() > bestQ) { bestQ = cell.leftQuality(); bestAction = "LEFT"; }
    if (cell.rightQuality() > bestQ) { bestQ = cell.rightQuality(); bestAction = "RIGHT"; }
    
    bestActionLabel_->setText(QString("⭐ Best Action: %1 (Q=%2)")
      .arg(bestAction)
      .arg(bestQ, 0, 'f', 2));
  }
  
  void clearInfo() {
    positionLabel_->setText("📍 Position: -");
    typeLabel_->setText("🏷️ Type: -");
    rewardLabel_->setText("💰 Reward: -");
    upArrow_->setArrow(0, -90, "UP");
    downArrow_->setArrow(0, 90, "DOWN");
    leftArrow_->setArrow(0, 180, "LEFT");
    rightArrow_->setArrow(0, 0, "RIGHT");
    noMoveLabel_->setText("NO MOVE\n-");
    bestActionLabel_->setText("⭐ Best Action: -");
  }
  
private:
  QLabel* positionLabel_;
  QLabel* typeLabel_;
  QLabel* rewardLabel_;
  ArrowWidget* upArrow_;
  ArrowWidget* downArrow_;
  ArrowWidget* leftArrow_;
  ArrowWidget* rightArrow_;
  QLabel* noMoveLabel_;
  QLabel* bestActionLabel_;
};

// Grid visualization widget - auto fits to window size
class GridWidget : public QWidget {
  Q_OBJECT
  
public:
  GridWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMouseTracking(true);
    setMinimumSize(400, 300);
  }
  
  void setData(const QLearningData* data) {
    data_ = data;
    update();
  }
  
  void setInfoPanel(CellInfoPanel* panel) {
    infoPanel_ = panel;
  }
  
protected:
  // Calculate cell size to fit the widget
  void calculateCellSize() {
    if (!data_ || data_->rows == 0 || data_->cols == 0) return;
    
    int availableWidth = width();
    int availableHeight = height();
    
    // Calculate cell size to fill the widget
    cellWidth_ = availableWidth / data_->cols;
    cellHeight_ = availableHeight / data_->rows;
    
    // Ensure minimum size for readability
    if (cellWidth_ < MIN_CELL_SIZE || cellHeight_ < MIN_CELL_SIZE) {
      // Use minimum size and allow scrolling (handled by parent scroll area if needed)
      cellWidth_ = std::max(MIN_CELL_SIZE, cellWidth_);
      cellHeight_ = std::max(MIN_CELL_SIZE, cellHeight_);
    }
  }
  
  void drawArrow(QPainter& painter, int cx, int cy, int action, int size) {
    // Calculate arrow angle based on action
    double angle = 0;
    switch (action) {
      case ACTION_UP:    angle = -90; break;
      case ACTION_DOWN:  angle = 90; break;
      case ACTION_LEFT:  angle = 180; break;
      case ACTION_RIGHT: angle = 0; break;
      default: return; // No arrow for NoMove
    }
    
    painter.save();
    painter.translate(cx, cy);
    painter.rotate(angle);
    
    // Arrow color - white with black outline for visibility
    painter.setPen(QPen(Qt::black, 1));
    painter.setBrush(Qt::white);
    
    // Draw arrow
    int len = size * 0.4;
    int head = size * 0.25;
    
    QPointF points[3] = {
      QPointF(len, 0),
      QPointF(len - head, -head/2),
      QPointF(len - head, head/2)
    };
    painter.drawPolygon(points, 3);
    painter.drawLine(-len/2, 0, len - head, 0);
    
    painter.restore();
  }
  
  void paintEvent(QPaintEvent* event) override {
    Q_UNUSED(event);
    
    if (!data_) return;
    
    // Recalculate cell size on each paint to fit current widget size
    calculateCellSize();
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Draw cells
    for (int r = 0; r < data_->rows; ++r) {
      for (int c = 0; c < data_->cols; ++c) {
        QRect cellRect(c * cellWidth_, r * cellHeight_, cellWidth_, cellHeight_);
        
        // Fill cell with heatmap color
        painter.fillRect(cellRect, data_->getCellColor(r, c, false));
        
        // Draw border
        painter.setPen(QPen(Qt::gray, 1));
        painter.drawRect(cellRect);
        
        // Draw optimal action arrow if cell is large enough
        if (cellWidth_ >= 12 && cellHeight_ >= 12) {
          int best_action = data_->getBestAction(r, c);
          if (best_action != 4) { // Not NoMove
            int cx = cellRect.center().x();
            int cy = cellRect.center().y();
            int arrow_size = std::min(cellWidth_, cellHeight_) * 0.8;
            drawArrow(painter, cx, cy, best_action, arrow_size);
          }
        }
        
        // Draw small indicator for trap/bingo if cell is large enough
        if (cellWidth_ >= 20 && cellHeight_ >= 20) {
          const auto& cell = data_->grid[r][c];
          if (cell.cell_type != qlearning::NORMAL_CELL) {
            painter.setPen(Qt::white);
            painter.setFont(QFont("Arial", std::max(6, cellWidth_ / 3), QFont::Bold));
            QString text = (cell.cell_type == qlearning::BINGO_CELL) ? "G" : "X";
            painter.drawText(cellRect, Qt::AlignCenter, text);
          }
        }
      }
    }
    
    // Draw highlight for selected/hovered cell
    if (hoverRow_ >= 0 && hoverCol_ >= 0 && hoverRow_ < data_->rows && hoverCol_ < data_->cols) {
      QRect cellRect(hoverCol_ * cellWidth_, hoverRow_ * cellHeight_, cellWidth_, cellHeight_);
      painter.setPen(QPen(Qt::red, std::max(2, cellWidth_ / 10)));
      painter.drawRect(cellRect.adjusted(1, 1, -1, -1));
    }
  }
  
  void mouseMoveEvent(QMouseEvent* event) override {
    if (!data_) return;
    
    // Recalculate cell size in case widget resized
    calculateCellSize();
    
    int col = event->pos().x() / cellWidth_;
    int row = event->pos().y() / cellHeight_;
    
    if (row >= 0 && row < data_->rows && col >= 0 && col < data_->cols) {
      if (row != hoverRow_ || col != hoverCol_) {
        hoverRow_ = row;
        hoverCol_ = col;
        update();
        
        // Update info panel
        if (infoPanel_) {
          infoPanel_->updateCellInfo(row, col, data_->grid[row][col]);
        }
      }
    }
  }
  
  void resizeEvent(QResizeEvent* event) override {
    QWidget::resizeEvent(event);
    update(); // Repaint on resize to adjust cell sizes
  }
  
  void leaveEvent(QEvent* event) override {
    Q_UNUSED(event);
    hoverRow_ = -1;
    hoverCol_ = -1;
    update();
    if (infoPanel_) {
      infoPanel_->clearInfo();
    }
  }
  
private:
  const QLearningData* data_ = nullptr;
  CellInfoPanel* infoPanel_ = nullptr;
  int cellWidth_ = 10;
  int cellHeight_ = 10;
  int hoverRow_ = -1;
  int hoverCol_ = -1;
};

// Main window
class MainWindow : public QMainWindow {
  Q_OBJECT
  
public:
  MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
    setWindowTitle("Q-Learning Visualizer");
    resize(1400, 900);
    
    // Central widget
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(5);
    mainLayout->setContentsMargins(5, 5, 5, 5);
    
    // Toolbar layout
    QHBoxLayout* toolbarLayout = new QHBoxLayout();
    
    // Fullscreen button
    QPushButton* fullscreenBtn = new QPushButton("⛶ Fullscreen (F11)", this);
    fullscreenBtn->setToolTip("Toggle fullscreen mode");
    connect(fullscreenBtn, &QPushButton::clicked, this, &MainWindow::toggleFullscreen);
    toolbarLayout->addWidget(fullscreenBtn);
    
    toolbarLayout->addStretch();
    
    // Grid info label
    gridInfoLabel_ = new QLabel("Grid: -", this);
    gridInfoLabel_->setStyleSheet("QLabel { padding: 5px 10px; font-weight: bold; }");
    toolbarLayout->addWidget(gridInfoLabel_);
    
    mainLayout->addLayout(toolbarLayout);
    
    // Content area with grid and side panel
    QHBoxLayout* contentLayout = new QHBoxLayout();
    
    // Left side: Grid
    QVBoxLayout* leftLayout = new QVBoxLayout();
    
    // Legend
    QLabel* legendLabel = new QLabel(
      "Legend: <span style='background-color:#64C864;padding:2px;'>Green</span>=Bingo "
      "<span style='background-color:#C86464;padding:2px;'>Red</span>=Trap "
      "<span style='background-color:#F0F0C8;padding:2px;'>Blue-ish</span>=Normal | "
      "F11: Fullscreen | Hover: View details in right panel", this);
    legendLabel->setStyleSheet("QLabel { padding: 5px; background-color: #f5f5f5; border: 1px solid #ddd; }");
    leftLayout->addWidget(legendLabel);
    
    // Grid widget - fills the available space
    gridWidget_ = new GridWidget(this);
    leftLayout->addWidget(gridWidget_, 1); // Stretch factor 1 to fill space
    
    contentLayout->addLayout(leftLayout, 1);
    
    // Right side: Info panel
    infoPanel_ = new CellInfoPanel(this);
    gridWidget_->setInfoPanel(infoPanel_);
    contentLayout->addWidget(infoPanel_);
    
    mainLayout->addLayout(contentLayout, 1); // Stretch factor 1
    
    setCentralWidget(centralWidget);
    
    // Status bar
    statusBar()->showMessage("Ready");
    
    // Setup shortcuts
    setupShortcuts();
  }
  
  void setupShortcuts() {
    // F11 for fullscreen
    QShortcut* fullscreenShortcut = new QShortcut(QKeySequence("F11"), this);
    connect(fullscreenShortcut, &QShortcut::activated, this, &MainWindow::toggleFullscreen);
    
    // Escape to exit fullscreen
    QShortcut* exitFullscreenShortcut = new QShortcut(QKeySequence("Escape"), this);
    connect(exitFullscreenShortcut, &QShortcut::activated, this, [this]() {
      if (isFullScreen()) {
        showNormal();
      }
    });
  }
  
  bool loadData(const QString& filepath) {
    if (!data_.load(filepath)) {
      QMessageBox::critical(this, "Error", "Failed to load: " + filepath);
      return false;
    }
    
    gridWidget_->setData(&data_);
    gridInfoLabel_->setText(QString("Grid: %1 x %2").arg(data_.rows).arg(data_.cols));
    statusBar()->showMessage(QString("Loaded %1x%2 grid from %3").arg(data_.rows).arg(data_.cols).arg(filepath));
    return true;
  }
  
public slots:
  void toggleFullscreen() {
    if (isFullScreen()) {
      showNormal();
    } else {
      showFullScreen();
    }
  }
  
private:
  GridWidget* gridWidget_;
  CellInfoPanel* infoPanel_;
  QLabel* gridInfoLabel_;
  QLearningData data_;
};

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  
  MainWindow window;
  
  QString jsonPath = "./q_learning.json";
  if (argc > 1) {
    jsonPath = argv[1];
  }
  
  if (window.loadData(jsonPath)) {
    window.show();
    return app.exec();
  }
  
  return 1;
}

#include "qt_visualizer.moc"
