
#include "dag.h"

void BuildGraph() {
  auto node_a = std::make_shared<GraphNode>("A", 0.0);
  auto node_b = std::make_shared<GraphNode>("B", 0.0);
  auto node_c = std::make_shared<GraphNode>("C", 0.0);
  auto node_d = std::make_shared<GraphNode>("D", 1.0, 1.0);
  auto node_e = std::make_shared<GraphNode>("E", 0.0);

  LOG_INFO << "build dag.\n";

  AddNeighbor(node_a, node_b);
  AddNeighbor(node_b, node_e);
  AddNeighbor(node_b, node_c);
  AddNeighbor(node_c, node_d);

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
}

class BellmanUpdate {
public:
  BellmanUpdate() {}
  virtual ~BellmanUpdate() {}

private:
  DAG values_;
};

int main() { BuildGraph(); }