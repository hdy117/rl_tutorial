#pragma once

#include <exception>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#define LOG_INFO std::cout << __FILE__ << ":" << __LINE__ << ":"
#define LOG_ERROR std::cerr << __FILE__ << ":" << __LINE__ << ":"

// graph node
struct GraphNode;
using GraphNodePtr = std::shared_ptr<GraphNode>;

// for simplicity, we set a max number of neighbors for each node
const int kNodeMaxNeighbors = 50;

// define a graph node, which has a name, a value, and a list of neighbors
struct GraphNode {
  std::string node_name_{""};
  double node_val_{0.0};
  std::vector<GraphNodePtr> neighbors_;

  GraphNode(const std::string &node_name, double node_val)
      : node_name_(node_name), node_val_(node_val) {
    neighbors_.reserve(kNodeMaxNeighbors);
  }
};

// add a neighbor to a node, return true if success, false if fail
bool AddNeighbor(GraphNodePtr &node, GraphNodePtr &neighbor) {
  if (node.get() == nullptr) {
    LOG_ERROR << "error, node is nullptr" << std::endl;
    throw std::runtime_error("error, node is nullptr");
    return false;
  }

  if (neighbor.get() == nullptr) {
    LOG_ERROR << "error, neighbor is nullptr" << std::endl;
    throw std::runtime_error("error, neighbor is nullptr");
    return false;
  }

  // check if repeated neighbor name exists in the node's neighbors
  for (const auto &adj_node : node->neighbors_) {
    if (adj_node->node_name_ == neighbor->node_name_) {
      LOG_ERROR << "error, neighbor with name " << neighbor->node_name_
                << " already exists in node " << node->node_name_ << std::endl;
      return false;
    }
  }

  // add the neighbor to the node's neighbors
  node->neighbors_.push_back(neighbor);
  LOG_INFO << "add neighbor with name " << neighbor->node_name_ << " to node "
           << node->node_name_ << std::endl;

  return true;
}

bool RemoveNeighborByName(GraphNodePtr &node, const std::string &node_name) {
  // check
  if (node.get() == nullptr) {
    LOG_ERROR << "error, node is nullptr" << std::endl;
    throw std::runtime_error("error, node is nullptr");
    return false;
  }

  // find and remove the neighbor with the given name
  for (auto it = node->neighbors_.begin(); it != node->neighbors_.end(); ++it) {
    if ((*it)->node_name_ == node_name) {
      auto iter = node->neighbors_.erase(it);
      iter->reset();
      return true;
    }
  }

  // if not found, print error and return false
  LOG_ERROR << "error, neighbor with name " << node_name
            << " not found in node " << node->node_name_ << std::endl;
  return false;
}

// visit node
void VisitNode(const GraphNodePtr &node) {
  if (node.get() == nullptr) {
    LOG_ERROR << "got nullptr of node, visit nothing";
    return;
  }
  LOG_INFO << "node name:" << node->node_name_
           << ", node val:" << node->node_val_ << std::endl;
}

// print seperator
void PrintSeperator(const std::string str = "") {
  std::cout << "---------------" << str << "--------------" << std::endl;
}

using GraphNodes = std::vector<GraphNodePtr>;
using Path = std::vector<GraphNodePtr>;
using PathList = std::vector<Path>;

// one DAG (directed acyclic graph)
class DAG {
public:
  DAG() {}
  virtual ~DAG() {}

public:
  // add node to the DAG
  void SetStartNode(const GraphNodePtr &node) { start_node_ = node; }

  // get start node
  GraphNodePtr GetStartNode() { return start_node_; }

  // BFS
  void BFS() {
    // queue for bfs
    std::queue<GraphNodePtr> nodes_queue;

    nodes_queue.push(start_node_);
    while (!nodes_queue.empty()) {
      const auto &cur_node = nodes_queue.front();
      for (const auto &adj_node : cur_node->neighbors_) {
        nodes_queue.push(adj_node);
      }
      VisitNode(cur_node);
      nodes_queue.pop();
    }
  }

  // DFS
  void DFS(const GraphNodePtr &node) {
    VisitNode(node);
    for (const auto &adj_node : node->neighbors_) {
      DFS(adj_node);
    }
  }

  // find paths from start node to end node
  PathList FindPathFromTo(const GraphNodePtr &node_a,
                          const GraphNodePtr &node_b) {
    PathList path_list;

    // early check
    if (node_a.get() == nullptr || node_b.get() == nullptr) {
      LOG_ERROR << "error, node_a or node_b is nullptr" << std::endl;
      throw std::runtime_error("error, node_a or node_b is nullptr");
      return path_list;
    }

    // find all path node_a and node_b in the DAG
    Path path;
    FindPath(start_node_, node_a, node_b, path_list, path);

    return path_list;
  }

  // print a path
  void PrintPath(const Path &path) {
    LOG_INFO << "print path: \n";
    for (const auto &node : path) {
      LOG_INFO << "\t" << node->node_name_ << ", " << node->node_val_ << "\n";
    }
  }

  void FindPath(const GraphNodePtr &node, const GraphNodePtr &node_a,
                const GraphNodePtr &node_b, PathList &path_list, Path &path) {
    // stop recursion
    if (node->node_name_ == node_b->node_name_) {
      if (!path.empty() && path.front()->node_name_ == node_a->node_name_) {
        // push end node of path
        path.push_back(node);

        // save this path
        path_list.push_back(path);
      } else {
        LOG_ERROR << "error, path is empty or path's first node is not node_a, "
                     "invalid path"
                  << std::endl;
      }

      // return anyway
      return;
    }

    if (node->node_name_ == node_a->node_name_) {
      // push start node of path
      path.push_back(node);
    } else if (!path.empty()) {
      // push path node
      path.push_back(node);
    }

    // find recursively
    for (const auto &adj_node : node->neighbors_) {
      FindPath(adj_node, node_a, node_b, path_list, path);
      path.pop_back();
    }
  }

private:
  GraphNodes nodes_;
  GraphNodePtr start_node_;
};