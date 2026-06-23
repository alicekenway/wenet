#include "utils/thread_pool.h"

namespace wenet_sdk::internal {

ThreadPool::ThreadPool(int num_threads) {
  if (num_threads < 1) {
    num_threads = 1;
  }
  workers_.reserve(static_cast<size_t>(num_threads));
  for (int i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this] { WorkerLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void ThreadPool::Enqueue(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
  }
  cv_.notify_one();
}

void ThreadPool::WorkerLoop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

}  // namespace wenet_sdk::internal
