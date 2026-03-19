#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QFile>
#include <QMessageBox>
#include <QScrollArea>
#include <QStatusBar>
#include <QPushButton>
#include <QShortcut>
#include <QKeySequence>
#include <cmath>

#include <google/protobuf/util/json_util.h>
#include "q_learning.pb.h"

using google::protobuf::util::JsonStringToMessage;

// Action indices (matches q_learning.h)
const int ACTION_UP = 0;
const int ACTION_DOWN = 1;
const int ACTION_LEFT = 2;
const int ACTION_RIGHT = 3;

// Default cell size
const int DEFAULT_CELL_WIDTH = 20;
const int DEFAULT_CELL_HEIGHT = 20;
const int MIN_CELL_SIZE = 5;
const int MAX_CELL_SIZE = 100;

// Cell data structure
struct CellData {
  double reward = 0.0;
  std::vector<double> qualities; // [up, down, left, right, nomove]
  qlearning::CellType cell_type = qlearning::NORMAL_CELL;
  
  double upQuality() const { return qualities.size() > ACTION_UP ? qualities[ACTION_UP] : 0.0; }
  double downQuality() const { return qualities.size() > ACTION_DOWN ? qualities[ACTION_DOWN] : 0.0; }
  double leftQuality() const { return qualities.size() > ACTION_LEFT ? qualities[ACTION_LEFT] : 0.0; }
  double rightQuality() const { return qualities.size() > ACTION_RIGHT ? qualities[ACTION_RIGHT] : 0.0; }
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
  
  QColor getCellColor(int r, int c) const {
    const auto& cell = grid[r][c];
    if (cell.cell_type == qlearning::BINGO_CELL) {
      return QColor(100, 200, 100); // Green for bingo
    } else if (cell.cell_type == qlearning::TRAP_CELL) {
      return QColor(200, 100, 100); // Red for trap
    }
    
    // Normal cell - shade based on max quality
    double max_q = -1e9;
    for (double q : cell.qualities) {
      max_q = std::max(max_q, q);
    }
    
    // Normalize to 0-255 for blue intensity
    int intensity = 200;
    if (max_q > -1000) {
      intensity = std::min(255, std::max(100, 200 + (int)(max_q * 50)));
    }
    return QColor(240, 240, intensity);
  }
};

// Grid visualization widget
class GridWidget : public QWidget {
  Q_OBJECT
  
public:
  GridWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setMouseTracking(true);
    setMinimumSize(800, 600);
  }
  
  void setData(const QLearningData* data) {
    data_ = data;
    resetZoom();
  }
  
  void setHoverInfoLabel(QLabel* label) {
    hoverLabel_ = label;
  }
  
  void resetZoom() {
    if (!data_) return;
    cellWidth_ = DEFAULT_CELL_WIDTH;
    cellHeight_ = DEFAULT_CELL_HEIGHT;
    updateSize();
  }
  
  void zoomIn() {
    cellWidth_ = std::min(MAX_CELL_SIZE, cellWidth_ + 3);
    cellHeight_ = std::min(MAX_CELL_SIZE, cellHeight_ + 3);
    updateSize();
  }
  
  void zoomOut() {
    cellWidth_ = std::max(MIN_CELL_SIZE, cellWidth_ - 3);
    cellHeight_ = std::max(MIN_CELL_SIZE, cellHeight_ - 3);
    updateSize();
  }
  
  double getZoomLevel() const {
    return static_cast<double>(cellWidth_) / DEFAULT_CELL_WIDTH;
  }

protected:
  void updateSize() {
    if (!data_) return;
    setMinimumSize(data_->cols * cellWidth_, data_->rows * cellHeight_);
    update();
    emit zoomChanged(getZoomLevel());
  }
  
  void paintEvent(QPaintEvent* event) override {
    Q_UNUSED(event);
    
    if (!data_) return;
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Draw cells
    for (int r = 0; r < data_->rows; ++r) {
      for (int c = 0; c < data_->cols; ++c) {
        QRect cellRect(c * cellWidth_, r * cellHeight_, cellWidth_, cellHeight_);
        
        // Fill cell
        painter.fillRect(cellRect, data_->getCellColor(r, c));
        
        // Draw border
        painter.setPen(QPen(Qt::gray, 1));
        painter.drawRect(cellRect);
        
        // Draw reward in small text if cell is large enough
        if (cellWidth_ > 30 && cellHeight_ > 30) {
          const auto& cell = data_->grid[r][c];
          if (std::abs(cell.reward) > 0.01) {
            painter.setPen(Qt::black);
            painter.setFont(QFont("Arial", 8));
            QString text = QString::number(cell.reward, 'f', 0);
            painter.drawText(cellRect, Qt::AlignCenter, text);
          }
        }
      }
    }
    
    // Draw hover overlay if mouse is over a cell
    if (hoverRow_ >= 0 && hoverCol_ >= 0 && hoverRow_ < data_->rows && hoverCol_ < data_->cols) {
      drawHoverOverlay(painter, hoverRow_, hoverCol_);
    }
  }
  
  void drawHoverOverlay(QPainter& painter, int r, int c) {
    const auto& cell = data_->grid[r][c];
    
    // Highlight current cell
    QRect cellRect(c * cellWidth_, r * cellHeight_, cellWidth_, cellHeight_);
    painter.setPen(QPen(Qt::red, 2));
    painter.drawRect(cellRect.adjusted(1, 1, -1, -1));
    
    // Draw arrows for each direction
    int cx = c * cellWidth_ + cellWidth_ / 2;
    int cy = r * cellHeight_ + cellHeight_ / 2;
    int arrowLen = std::min(cellWidth_, cellHeight_) / 3;
    int offset = arrowLen + 5;
    
    // Arrow positions (up, down, left, right)
    struct ArrowInfo {
      int x, y;
      double angle;
      double quality;
      QString label;
    };
    
    std::vector<ArrowInfo> arrows = {
      {cx, cy - offset, -90, cell.upQuality(), "U"},
      {cx, cy + offset, 90, cell.downQuality(), "D"},
      {cx - offset, cy, 180, cell.leftQuality(), "L"},
      {cx + offset, cy, 0, cell.rightQuality(), "R"}
    };
    
    for (const auto& arrow : arrows) {
      drawArrow(painter, arrow.x, arrow.y, arrow.angle, arrow.quality, arrow.label);
    }
    
    // Draw reward in center
    painter.setPen(Qt::darkRed);
    painter.setFont(QFont("Arial", 10, QFont::Bold));
    QString rewardText = QString("R:%1").arg(cell.reward, 0, 'f', 1);
    QRect textRect(cx - cellWidth_/2 + 2, cy - 8, cellWidth_ - 4, 16);
    painter.drawText(textRect, Qt::AlignCenter, rewardText);
  }
  
  void drawArrow(QPainter& painter, int x, int y, double angle, double quality, const QString& label) {
    painter.save();
    painter.translate(x, y);
    painter.rotate(angle);
    
    // Arrow color based on quality
    QColor color = getQualityColor(quality);
    painter.setPen(QPen(color, 2));
    painter.setBrush(color);
    
    // Arrow shaft
    int len = std::max(8, std::min(20, std::min(cellWidth_, cellHeight_) / 2));
    painter.drawLine(-len/2, 0, len/2, 0);
    
    // Arrow head
    QPointF points[3] = {
      QPointF(len/2, 0),
      QPointF(len/2 - 5, -4),
      QPointF(len/2 - 5, 4)
    };
    painter.drawPolygon(points, 3);
    
    // Quality value
    painter.rotate(-angle); // Reset rotation for text
    painter.setPen(Qt::black);
    painter.setFont(QFont("Arial", 7));
    QString valText = QString::number(quality, 'f', 2);
    QRect textRect(-20, -20, 40, 15);
    painter.drawText(textRect, Qt::AlignCenter, valText);
    
    // Direction label
    QRect labelRect(-10, 5, 20, 12);
    painter.setPen(color.darker());
    painter.drawText(labelRect, Qt::AlignCenter, label);
    
    painter.restore();
  }
  
  QColor getQualityColor(double q) const {
    if (q > 0) {
      int g = std::min(255, (int)(100 + q * 100));
      return QColor(0, g, 0);
    } else if (q < -500) {
      return QColor(150, 0, 0); // Dark red for very negative
    } else {
      int r = std::min(255, (int)(100 - q * 10));
      return QColor(r, 0, 0);
    }
  }
  
  void mouseMoveEvent(QMouseEvent* event) override {
    if (!data_) return;
    
    int col = event->pos().x() / cellWidth_;
    int row = event->pos().y() / cellHeight_;
    
    if (row != hoverRow_ || col != hoverCol_) {
      hoverRow_ = row;
      hoverCol_ = col;
      update();
      
      // Update hover info label
      if (hoverLabel_ && row >= 0 && row < data_->rows && col >= 0 && col < data_->cols) {
        const auto& cell = data_->grid[row][col];
        QString info = QString("Cell[%1,%2] Reward:%3 | U:%4 D:%5 L:%6 R:%7")
          .arg(row).arg(col)
          .arg(cell.reward, 0, 'f', 2)
          .arg(cell.upQuality(), 0, 'f', 2)
          .arg(cell.downQuality(), 0, 'f', 2)
          .arg(cell.leftQuality(), 0, 'f', 2)
          .arg(cell.rightQuality(), 0, 'f', 2);
        hoverLabel_->setText(info);
      }
    }
  }
  
  void wheelEvent(QWheelEvent* event) override {
    if (!data_) return;
    
    // Zoom with mouse wheel
    if (event->angleDelta().y() > 0) {
      zoomIn();
    } else {
      zoomOut();
    }
    event->accept();
  }
  
  void leaveEvent(QEvent* event) override {
    Q_UNUSED(event);
    hoverRow_ = -1;
    hoverCol_ = -1;
    update();
    if (hoverLabel_) {
      hoverLabel_->setText("Hover over a cell to see details");
    }
  }
  
signals:
  void zoomChanged(double level);
  
private:
  const QLearningData* data_ = nullptr;
  int cellWidth_ = DEFAULT_CELL_WIDTH;
  int cellHeight_ = DEFAULT_CELL_HEIGHT;
  int hoverRow_ = -1;
  int hoverCol_ = -1;
  QLabel* hoverLabel_ = nullptr;
};

// Main window
class MainWindow : public QMainWindow {
  Q_OBJECT
  
public:
  MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
    setWindowTitle("Q-Learning Visualizer");
    resize(1200, 800);
    
    // Central widget with layout
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
    
    // Zoom out button
    QPushButton* zoomOutBtn = new QPushButton("🔍- Zoom Out", this);
    zoomOutBtn->setToolTip("Zoom out (Mouse wheel down)");
    connect(zoomOutBtn, &QPushButton::clicked, this, [this]() {
      if (gridWidget_) gridWidget_->zoomOut();
    });
    toolbarLayout->addWidget(zoomOutBtn);
    
    // Reset zoom button
    QPushButton* resetZoomBtn = new QPushButton("⟲ Reset Zoom (100%)", this);
    resetZoomBtn->setToolTip("Reset to normal size (Ctrl+0)");
    resetZoomBtn->setStyleSheet("QPushButton { font-weight: bold; color: #0066cc; }");
    connect(resetZoomBtn, &QPushButton::clicked, this, &MainWindow::resetZoom);
    toolbarLayout->addWidget(resetZoomBtn);
    
    // Zoom in button
    QPushButton* zoomInBtn = new QPushButton("🔍+ Zoom In", this);
    zoomInBtn->setToolTip("Zoom in (Mouse wheel up)");
    connect(zoomInBtn, &QPushButton::clicked, this, [this]() {
      if (gridWidget_) gridWidget_->zoomIn();
    });
    toolbarLayout->addWidget(zoomInBtn);
    
    // Zoom level label
    zoomLabel_ = new QLabel("Zoom: 100%", this);
    zoomLabel_->setStyleSheet("QLabel { padding: 5px 10px; font-weight: bold; }");
    toolbarLayout->addWidget(zoomLabel_);
    
    toolbarLayout->addStretch();
    mainLayout->addLayout(toolbarLayout);
    
    // Info label
    infoLabel_ = new QLabel("Hover over a cell to see details | Mouse wheel to zoom | F11 for fullscreen", this);
    infoLabel_->setAlignment(Qt::AlignLeft);
    infoLabel_->setStyleSheet("QLabel { padding: 8px; background-color: #f0f0f0; border: 1px solid #ccc; }");
    mainLayout->addWidget(infoLabel_);
    
    // Legend
    QLabel* legendLabel = new QLabel(
      "Legend: <span style='background-color:#64C864;padding:2px;'>Green</span>=Bingo "
      "<span style='background-color:#C86464;padding:2px;'>Red</span>=Trap "
      "<span style='background-color:#F0F0C8;padding:2px;'>Blue-ish</span>=Normal "
      "| Arrows: ↑=Up ↓=Down ←=Left →=Right (color=quality value)", this);
    legendLabel->setStyleSheet("QLabel { padding: 5px; }");
    mainLayout->addWidget(legendLabel);
    
    // Grid widget in scroll area
    gridWidget_ = new GridWidget(this);
    gridWidget_->setHoverInfoLabel(infoLabel_);
    connect(gridWidget_, &GridWidget::zoomChanged, this, &MainWindow::updateZoomLabel);
    
    QScrollArea* scrollArea = new QScrollArea(this);
    scrollArea->setWidget(gridWidget_);
    scrollArea->setWidgetResizable(true);
    mainLayout->addWidget(scrollArea);
    
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
    
    // Ctrl+0 for reset zoom
    QShortcut* resetZoomShortcut = new QShortcut(QKeySequence("Ctrl+0"), this);
    connect(resetZoomShortcut, &QShortcut::activated, this, &MainWindow::resetZoom);
    
    // Ctrl++ for zoom in
    QShortcut* zoomInShortcut = new QShortcut(QKeySequence("Ctrl++"), this);
    connect(zoomInShortcut, &QShortcut::activated, this, [this]() {
      if (gridWidget_) gridWidget_->zoomIn();
    });
    
    // Ctrl+- for zoom out
    QShortcut* zoomOutShortcut = new QShortcut(QKeySequence("Ctrl+-"), this);
    connect(zoomOutShortcut, &QShortcut::activated, this, [this]() {
      if (gridWidget_) gridWidget_->zoomOut();
    });
    
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
    statusBar()->showMessage(QString("Loaded %1x%2 grid").arg(data_.rows).arg(data_.cols));
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
  
  void resetZoom() {
    if (gridWidget_) {
      gridWidget_->resetZoom();
    }
  }
  
  void updateZoomLabel(double level) {
    zoomLabel_->setText(QString("Zoom: %1%").arg(static_cast<int>(level * 100)));
  }
  
private:
  GridWidget* gridWidget_;
  QLabel* infoLabel_;
  QLabel* zoomLabel_;
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
