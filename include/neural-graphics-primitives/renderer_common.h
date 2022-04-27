/*
 * Copyright (c) Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_common.h
 *  @author Moonsik Park, Korea Institute of Science and Technology
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/renderer_common.h>
#include <neural-graphics-primitives/testbed.h>

#include <proto/nes.pb.h>

#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <exception>

class lock_timeout : public std::exception
{
    virtual const char *what() const throw()
    {
        return "Waiting for lock timed out.";
    }
};

template <class T>
class ThreadSafeQueue
{
private:
    unsigned int _max_size;
    std::queue<T> _queue;
    std::condition_variable _pusher, _popper;
    std::mutex _mutex;

    using unique_lock = std::unique_lock<std::mutex>;

public:
    ThreadSafeQueue(unsigned int max_size) : _max_size(max_size) {}

    template <class U>
    void push(U &&item)
    {
        unique_lock lock(this->_mutex);
        if (this->_pusher.wait_for(lock, std::chrono::milliseconds(10000), [&]
                                   { return this->_queue.size() < this->_max_size; }))
        {
            this->_queue.push(std::forward<U>(item));
            this->_popper.notify_one();
        }
        else
        {
            throw lock_timeout{};
        }
    }

    T pop()
    {
        unique_lock lock(this->_mutex);
        if (this->_popper.wait_for(lock, std::chrono::milliseconds(10000), [&]
                                   { return this->_queue.size() > 0; }))
        {
            T item = std::move(this->_queue.front());
            this->_queue.pop();
            this->_pusher.notify_one();
            return item;
        }
        else
        {
            throw lock_timeout{};
        }
    }
};

namespace nes
{
    std::string render(ngp::Testbed &testbed, nesproto::FrameRequest request, ngp::ERenderMode mode, float *fbuf, char *cbuf);
    Eigen::Matrix<float, 3, 4> cam_to_matrix(nesproto::Camera cam);
    int socket_send_blocking(int clientfd, uint8_t *buf, size_t size);
    int socket_send_blocking_lpf(int clientfd, uint8_t *buf, size_t size);
    int socket_receive_blocking(int clientfd, uint8_t *buf, size_t size);
    std::string socket_receive_blocking_lpf(int clientfd);
}
