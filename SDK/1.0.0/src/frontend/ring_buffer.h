#ifndef WENET_SDK_SRC_FRONTEND_RING_BUFFER_H_
#define WENET_SDK_SRC_FRONTEND_RING_BUFFER_H_

#include <cstddef>
#include <vector>

namespace wenet_sdk::internal {

class FloatRingBuffer {
 public:
  explicit FloatRingBuffer(size_t capacity);

  size_t Write(const float* data, size_t n);
  size_t Read(float* data, size_t n);
  void Clear();
  size_t size() const { return size_; }
  size_t capacity() const { return buffer_.size(); }

 private:
  std::vector<float> buffer_;
  size_t read_pos_ = 0;
  size_t write_pos_ = 0;
  size_t size_ = 0;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_FRONTEND_RING_BUFFER_H_
