#pragma once
#include <atomic>
#include <cassert>
#include <sstream>
#include <thread>

namespace utils {
static inline size_t get_thread_id() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  return atoi(ss.str().c_str());
}
static thread_local size_t g_threadId = get_thread_id();

class SpinMutex {
public:
  SpinMutex() = default;
  SpinMutex(const SpinMutex &) = delete;
  SpinMutex &operator=(const SpinMutex &) = delete;
  inline void lock() noexcept {
    while (_flag.exchange(true, std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    _owner = g_threadId;
  }

  inline bool try_lock() noexcept {
    bool b = !_flag.load(std::memory_order_relaxed) &&
             !_flag.exchange(true, std::memory_order_acquire);
    if (b)
      _owner = g_threadId;
    return b;
  }

  inline void unlock() noexcept {
    assert(_owner == g_threadId);
    _owner = 0;
    _flag.store(false, std::memory_order_release);
  }

  inline bool is_locked() const {
    return _flag.load(std::memory_order_relaxed);
  }

  inline size_t owner() const { return _owner; }

protected:
  std::atomic<bool> _flag = ATOMIC_VAR_INIT(false);
  size_t _owner = 0;
};

class SharedSpinMutex {
public:
  SharedSpinMutex() = default;
  SharedSpinMutex(const SharedSpinMutex &) = delete;
  SharedSpinMutex &operator=(const SharedSpinMutex &) = delete;

  inline void lock() noexcept {
    while (_writeFlag.exchange(true, std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    while (_readCount.load(std::memory_order_relaxed) > 0) {
      std::this_thread::yield();
    }

    _owner = g_threadId;
  }

  inline bool try_lock() noexcept {
    if (_readCount.load(std::memory_order_relaxed) == 0 &&
        !_writeFlag.exchange(true, std::memory_order_acquire)) {
      if (_readCount.load(std::memory_order_relaxed) > 0) {
        _writeFlag.store(false, std::memory_order_relaxed);
        return false;
      }

      _owner = g_threadId;
      return true;
    }

    return false;
  }

  void unlock() noexcept {
    assert(_owner == g_threadId);
    _owner = 0;
    _writeFlag.store(false, std::memory_order_release);
  }

  inline void lock_shared() noexcept {
    while (true) {
      _readCount.fetch_add(1, std::memory_order_acquire);
      if (!_writeFlag.load(std::memory_order_relaxed))
        break;

      _readCount.fetch_sub(1, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  }

  inline bool try_lock_shared() noexcept {
    if (!_writeFlag.load(std::memory_order_relaxed)) {
      _readCount.fetch_add(1, std::memory_order_acquire);
      if (_writeFlag.load(std::memory_order_relaxed)) {
        _readCount.fetch_sub(1, std::memory_order_relaxed);
        return false;
      }

      return true;
    }

    return false;
  }

  inline void unlock_shared() noexcept {
    assert(_owner == 0);
    _readCount.fetch_sub(1, std::memory_order_relaxed);
  }

  inline bool is_write_locked() const {
    return _writeFlag.load(std::memory_order_relaxed);
  }

  inline uint32_t read_locked_count() const {
    return _readCount.load(std::memory_order_relaxed);
  }

  inline bool is_locked() const {
    return _writeFlag.load(std::memory_order_relaxed) ||
           _readCount.load(std::memory_order_relaxed) > 0;
  }

protected:
  std::atomic<int32_t> _readCount{0};
  std::atomic<bool> _writeFlag{false};
  size_t _owner = 0;
};

class ReentrantSpinMutex {
public:
  ReentrantSpinMutex() = default;
  ReentrantSpinMutex(const ReentrantSpinMutex &) = delete;
  ReentrantSpinMutex &operator=(const ReentrantSpinMutex &) = delete;

  inline void lock() noexcept {
    if (_owner == g_threadId) {
      _reenCount++;
      return;
    }

    while (_flag.exchange(true, std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    assert(_reenCount == 0 && _owner == 0);
    _owner = g_threadId;
    _reenCount = 1;
  }

  bool try_lock() noexcept {
    size_t currId = g_threadId;
    if (_owner == currId) {
      _reenCount++;
      return true;
    }

    if (!_flag.load(std::memory_order_relaxed) &&
        !_flag.exchange(true, std::memory_order_acquire)) {
      assert(_reenCount == 0);
      _owner = currId;
      _reenCount = 1;
      return true;
    }

    return false;
  }

  inline void unlock() noexcept {
    assert(_owner == g_threadId);
    _reenCount--;
    if (_reenCount == 0) {
      _owner = 0;
      _flag.store(false, std::memory_order_release);
    }
  }

  inline bool is_locked() const {
    return _flag.load(std::memory_order_relaxed);
  }

  inline int32_t reentrant_count() const { return _reenCount; }

protected:
  std::atomic<bool> _flag = ATOMIC_VAR_INIT(false);
  size_t _owner = 0;
  int32_t _reenCount = 0;
};

class ReentrantSharedSpinMutex {
public:
  ReentrantSharedSpinMutex() = default;
  ReentrantSharedSpinMutex(const ReentrantSharedSpinMutex &) = delete;
  ReentrantSharedSpinMutex &
  operator=(const ReentrantSharedSpinMutex &) = delete;

  inline void lock() noexcept {
    size_t currId = g_threadId;
    if (_owner == currId) {
      assert(_reenCount > 0);
      _reenCount++;
      return;
    }

    while (_writeFlag.exchange(true, std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    while (_readCount.load(std::memory_order_relaxed) > 0) {
      std::this_thread::yield();
    }

    assert(_reenCount == 0);
    _owner = currId;
    _reenCount = 1;
  }

  inline bool try_lock() noexcept {
    size_t currId = g_threadId;
    if (_owner == currId) {
      assert(_reenCount > 0);
      _reenCount++;
      return true;
    }

    if (_readCount.load(std::memory_order_relaxed) == 0 &&
        !_writeFlag.exchange(true, std::memory_order_acquire)) {
      if (_readCount.load(std::memory_order_relaxed) > 0) {
        _writeFlag.store(false, std::memory_order_relaxed);
        return false;
      }

      assert(_reenCount == 0);
      _owner = currId;
      _reenCount = 1;
      return true;
    }

    return false;
  }

  void unlock() noexcept {
    assert(_owner == g_threadId);
    _reenCount--;
    if (_reenCount == 0) {
      _owner = 0;
      _writeFlag.store(false, std::memory_order_release);
    }
  }

  inline void lock_shared() noexcept {
    while (true) {
      _readCount.fetch_add(1, std::memory_order_acquire);
      if (!_writeFlag.load(std::memory_order_relaxed))
        break;

      _readCount.fetch_sub(1, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  }

  inline bool try_lock_shared() noexcept {
    if (!_writeFlag.load(std::memory_order_relaxed)) {
      _readCount.fetch_add(1, std::memory_order_acquire);
      if (_writeFlag.load(std::memory_order_relaxed)) {
        _readCount.fetch_sub(1, std::memory_order_relaxed);
        return false;
      }

      return true;
    }

    return false;
  }

  inline void unlock_shared() noexcept {
    _readCount.fetch_sub(1, std::memory_order_relaxed);
  }

  inline bool is_write_locked() const {
    return _writeFlag.load(std::memory_order_relaxed);
  }

  inline uint32_t read_locked_count() const {
    return _readCount.load(std::memory_order_relaxed);
  }

  inline bool is_locked() const {
    return _writeFlag.load(std::memory_order_relaxed) ||
           _readCount.load(std::memory_order_relaxed) > 0;
  }

  inline int32_t reentrant_count() const { return _reenCount; }

protected:
  std::atomic<int32_t> _readCount{0};
  std::atomic<bool> _writeFlag{false};
  size_t _owner = 0;
  int32_t _reenCount = 0;
};

// class SpinMutex {
// public:
//  SpinMutex() = default;
//  SpinMutex(const SpinMutex&) = delete;
//  SpinMutex& operator=(const SpinMutex&) = delete;
//  inline void lock() noexcept {
//  }
//
//  inline bool try_lock() noexcept {
//    return true;
//  }
//
//  inline void unlock() noexcept {
//  }
//
//  inline bool is_locked() const {
//    return true;
//  }
//
//  inline size_t owner() const { return _owner; }
//
// protected:
//  std::atomic<bool> _flag = ATOMIC_VAR_INIT(false);
//  size_t _owner = 0;
//};
//
// class SharedSpinMutex {
// public:
//  SharedSpinMutex() = default;
//  SharedSpinMutex(const SharedSpinMutex&) = delete;
//  SharedSpinMutex& operator=(const SharedSpinMutex&) = delete;
//
//  inline void lock() noexcept {
//  }
//
//  inline bool try_lock() noexcept {
//    return true;
//  }
//
//  void unlock() noexcept {
//  }
//
//  inline void lock_shared() noexcept {
//  }
//
//  inline bool try_lock_shared() noexcept {
//    return true;
//  }
//
//  inline void unlock_shared() noexcept {
//  }
//
//  inline bool is_write_locked() const {
//    return true;
//  }
//
//  inline uint32_t read_locked_count() const {
//    return 0;
//  }
//
//  inline bool is_locked() const {
//    return true;
//  }
//
// protected:
//  std::atomic<int32_t> _readCount{ 0 };
//  std::atomic<bool> _writeFlag{ false };
//  size_t _owner = 0;
//};
//
// class ReentrantSpinMutex {
// public:
//  ReentrantSpinMutex() = default;
//  ReentrantSpinMutex(const ReentrantSpinMutex&) = delete;
//  ReentrantSpinMutex& operator=(const ReentrantSpinMutex&) = delete;
//
//  inline void lock() noexcept {
//  }
//
//  bool try_lock() noexcept {
//    return true;
//  }
//
//  inline void unlock() noexcept {
//  }
//
//  inline bool is_locked() const {
//    return true;
//  }
//
//  inline int32_t reentrant_count() const { return _reenCount; }
//
// protected:
//  std::atomic<bool> _flag = ATOMIC_VAR_INIT(false);
//  size_t _owner = 0;
//  int32_t _reenCount = 0;
//};
//
// class ReentrantSharedSpinMutex {
// public:
//  ReentrantSharedSpinMutex() = default;
//  ReentrantSharedSpinMutex(const ReentrantSharedSpinMutex&) = delete;
//  ReentrantSharedSpinMutex&
//    operator=(const ReentrantSharedSpinMutex&) = delete;
//
//  inline void lock() noexcept {
//  }
//
//  inline bool try_lock() noexcept {
//    return true;
//  }
//
//  void unlock() noexcept {
//  }
//
//  inline void lock_shared() noexcept {
//  }
//
//  inline bool try_lock_shared() noexcept {
//    return true;
//  }
//
//  inline void unlock_shared() noexcept {
//  }
//
//  inline bool is_write_locked() const {
//    return true;
//  }
//
//  inline uint32_t read_locked_count() const {
//    return 0;
//  }
//
//  inline bool is_locked() const {
//    return true;
//  }
//
//  inline int32_t reentrant_count() const { return _reenCount; }
//
// protected:
//  std::atomic<int32_t> _readCount{ 0 };
//  std::atomic<bool> _writeFlag{ false };
//  size_t _owner = 0;
//  int32_t _reenCount = 0;
//};
} // namespace utils
