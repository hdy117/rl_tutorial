
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

// bellman update dag, the correct version
class BellmanDAG : public DAG {
public:
  BellmanDAG(double alpha = 0.001, double gamma = 0.9, int max_epoches = 3000)
      : alpha_(alpha), gamma_(gamma), max_epoches_(max_epoches) {}

public:
  void BellmanUpdate(const GraphNodePtr &final_node) {
    for (auto i = 0; i < max_epoches_; ++i) {
      MaxValue(start_node_, final_node);
    }
  }

  double MaxValue(const GraphNodePtr &node, const GraphNodePtr &final_node) {
    // stop scenario 1
    if (node->neighbors_.size() == 0) {
      return node->value_;
    }

    // stop scenario 2
    if (node->node_name_ == final_node->node_name_) {
      return final_node->value_;
    }

    double max_value = -1e9;
    for (const auto &adj_node : node->neighbors_) {
      max_value = std::max(max_value, MaxValue(adj_node, final_node));
    }

    // bellman update
    double value_target = node->reward_ + gamma_ * max_value;
    double dt_error = value_target - node->value_;
    node->value_ = node->value_ + alpha_ * dt_error;

    // return
    return node->value_;
  }

public:
  void Pi(const GraphNodePtr &cur_node, const GraphNodePtr &target_node) {
    LOG_INFO << "current node:" << cur_node->node_name_
             << ", value:" << cur_node->value_ << "\n";
    // final case
    if (target_node->node_name_ == cur_node->node_name_) {
      LOG_INFO << "bingo reach target.\n";
      return;
    }

    // climp one step
    GraphNodePtr climp_next_node;
    double max_value = -1e9;
    for (const auto &adj_node : cur_node->neighbors_) {
      if (adj_node->value_ > max_value) {
        max_value = adj_node->value_;
        climp_next_node = adj_node;
      }
    }

    // go on
    Pi(climp_next_node, target_node);
  }

private:
  double alpha_ = 0.001;   //  learning rate
  double gamma_ = 0.9;     // how much to trust the future
  int max_epoches_ = 3000; // max epoches
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
  auto dag = BellmanDAG(0.001, 0.9, 6000);
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
  PrintSeperator("not whole correct bellman update");
  BellmanUpdate bellman_update{0.001, 0.9};
  const int kBellmanUpadteEpochs = 1000;

  // not the correct version, since V(a) depends on max(V(b),V(c))
  // for (const auto &path : path_list) {
  //   for (auto i = 0; i < kBellmanUpadteEpochs; i++) {
  //     bellman_update.BellmanUpdateFunc(path);
  //   }
  //   dag.PrintPath(path);
  // }

  // the correct version of bellman update
  PrintSeperator("correct bellman update");
  dag.BellmanUpdate(node_d);
  PrintSeperator("DAG BFS value after bellman update");
  dag.BFS();
  PrintSeperator("policy using bellman update");
  dag.Pi(node_a, node_d);
}

int main() { BuildGraph(); }