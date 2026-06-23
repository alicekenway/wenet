#ifndef WENET_SDK_SRC_UTILS_THREAD_POOL_H_
#define WENET_SDK_SRC_UTILS_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace wenet_sdk::internal {

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);
  ~ThreadPool();

  void Enqueue(std::function<void()> task);

 private:
  void WorkerLoop();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_UTILS_THREAD_POOL_H_
