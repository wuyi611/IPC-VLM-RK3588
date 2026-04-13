#ifndef RKNN_YOLOV5_DEMO_PIPELINE_BOUNDED_QUEUE_H_
#define RKNN_YOLOV5_DEMO_PIPELINE_BOUNDED_QUEUE_H_

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

// 文件说明：
// 提供一个线程安全的有界阻塞队列，作为不同流水线阶段之间的数据通道。

namespace rknn_demo {

// `BoundedQueue` 是一个带容量上限的线程安全队列。
// 这个项目优先追求低延迟，因此当队列已满时不会阻塞生产者，
// 而是丢弃最旧的数据，再插入最新的数据。
template <typename T>
class BoundedQueue {
public:
    // 使用给定容量创建队列。
    // `capacity == 0` 时可理解为不限制容量。
    explicit BoundedQueue(size_t capacity) : capacity_(capacity), stopped_(false) {}

    // 向队列尾部压入一个元素。
    // `dropped_oldest` 用于告诉调用方本次是否因为队列已满而丢弃了最旧元素。
    // 返回 `false` 表示队列已经停止，不再接受新数据。
    bool push(T item, bool *dropped_oldest) {
        // 进入临界区，保护队列状态和停止标记。
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopped_) {
            return false;
        }

        // 记录本次写入是否触发了“丢最旧元素”的策略。
        bool dropped = false;
        if (capacity_ != 0 && queue_.size() >= capacity_) {
            queue_.pop_front();
            dropped = true;
        }

        // 把最新数据放入队尾，然后唤醒一个等待中的消费者。
        queue_.push_back(std::move(item));
        lock.unlock();
        cond_.notify_one();

        if (dropped_oldest != NULL) {
            *dropped_oldest = dropped;
        }
        return true;
    }

    // 从队列头部取出一个元素。
    // 当队列为空时会阻塞等待；当队列已停止且没有剩余数据时返回 `false`。
    bool pop(T *item) {
        // 进入临界区，等待生产者投递数据或队列停止。
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]() { return stopped_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }

        *item = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    // 停止队列。
    // 停止后所有等待中的消费者都会被唤醒，后续 `push()` 会失败。
    void stop() {
        // 保护停止状态写入，避免与并发 `push()` / `pop()` 竞争。
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cond_.notify_all();
    }

private:
    // 队列最大容量。
    size_t capacity_;

    // 队列是否已经进入停止状态。
    bool stopped_;

    // 实际存储数据的双端队列。
    std::deque<T> queue_;

    // 保护队列和停止状态的互斥锁。
    std::mutex mutex_;

    // 用于协调生产者和消费者之间的等待与唤醒。
    std::condition_variable cond_;
};

}  // namespace rknn_demo

#endif  // RKNN_YOLOV5_DEMO_PIPELINE_BOUNDED_QUEUE_H_
