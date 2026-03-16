
#include "dag.h"

class BellmanUpdate {
public:
  BellmanUpdate(double alpha = 0.001, double gamma = 0.9)
      : gamma_(gamma), alpha_(alpha) {}
  virtual ~BellmanUpdate() {}

public:
  void BellmanUpdateFunc(const Path path) {
    int64_t last_index = static_cast<int64_t>(path.size()) - 1;
    for (auto i = last_index; i >= 1; i--) {
      // get node
      auto node_cur = path.at(i - 1);
      auto node_next = path.at(i);

      double value_target = node_cur->reward_ + gamma_ * node_next->value_;
      double dt_error = value_target - node_cur->value_;
      node_cur->value_ = node_cur->value_ + alpha_ * dt_error;
    }
  }

private:
  double alpha_; // learning rate
  double gamma_; // gamma, value ratio of future
};

void BuildGraph() {
  auto node_a = std::make_shared<GraphNode>("A", 0.0);
  auto node_b = std::make_shared<GraphNode>("B", 0.0);
  auto node_c = std::make_shared<GraphNode>("C", 0.0);
  auto node_d = std::make_shared<GraphNode>("D", 1.0, 1.0);
  auto node_e = std::make_shared<GraphNode>("E", 0.0);
  auto node_f = std::make_shared<GraphNode>("F", 0.0);

  LOG_INFO << "build dag.\n";

  AddNeighbor(node_a, node_b);
  AddNeighbor(node_b, node_c);
  AddNeighbor(node_b, node_e);
  AddNeighbor(node_c, node_d);
  AddNeighbor(node_d, node_f);

  AddNeighbor(node_a, node_c);

  // new dag
  auto dag = DAG();
  dag.SetStartNode(node_a);

  // bfs
  PrintSeperator("BFS");
  dag.BFS();

  // dfs
  PrintSeperator("DFS");
  auto start_node = dag.GetStartNode();
  dag.DFS(start_node);

  // find path
  PrintSeperator("Path Finder");
  auto path_list = dag.FindPathFromTo(node_a, node_d);
  for (const auto &path : path_list) {
    dag.PrintPath(path);
  }

  // bellman update
  PrintSeperator("bellman update");
  BellmanUpdate bellman_update{0.001, 0.9};
  const int kBellmanUpadteEpochs = 1000;

  for (const auto &path : path_list) {
    for (auto i = 0; i < kBellmanUpadteEpochs; i++) {
      bellman_update.BellmanUpdateFunc(path);
    }
    dag.PrintPath(path);
  }
}

int main() { BuildGraph(); }