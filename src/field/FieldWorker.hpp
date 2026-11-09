#pragma once

#include "field/Parallel.hpp"
#include "field/Playground.hpp"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace fieldengine::field {

// One thread, one job at a time. Persistent, because OpenMP keeps its team per thread that starts it.
struct Playground::FieldWorker {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable changed;
    std::function<void()> job;
    bool busy = false;
    bool stop = false;

    FieldWorker()
    {
        thread = std::thread([this] {
            limit_threads();
            std::unique_lock<std::mutex> lock(mutex);
            while (true) {
                changed.wait(lock, [this] { return stop || busy; });
                if (stop) {
                    return;
                }
                lock.unlock();
                job();
                lock.lock();
                busy = false;
                changed.notify_all();
            }
        });
    }
    ~FieldWorker()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [this] { return !busy; });
            stop = true;
            changed.notify_all();
        }
        thread.join();
    }
    void submit(std::function<void()> next)
    {
        std::lock_guard<std::mutex> lock(mutex);
        job = std::move(next);
        busy = true;
        changed.notify_all();
    }
    [[nodiscard]] bool working()
    {
        const std::lock_guard<std::mutex> lock(mutex);
        return busy;
    }
    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [this] { return !busy; });
    }
};

} // namespace fieldengine::field
