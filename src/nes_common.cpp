/*
 * Copyright (c) 2020-2022, Moonsik Park.  All rights reserved.
 *
 */

/** @file   nes_common.cpp
 *  @author Moonsik Park, Korean Institute of Science and Technology
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nes_common.h>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace nes
{
    int socket_send_blocking(int clientfd, uint8_t *buf, size_t size)
    {
        ssize_t ret;
        ssize_t sent = 0;

        while (sent < size)
        {
            ret = send(clientfd, buf + sent, size - sent, MSG_NOSIGNAL);
            if (ret < 0)
            {
                // Buffer is full. Try again.
                if (errno == EAGAIN)
                {
                    continue;
                }
                // Misc error. Terminate the socket.
                tlog::error() << "socket_send_blocking: " << std::string(std::strerror(errno));
                return -errno;
            }
            sent += ret;
        }

        return 0;
    }

    // Send message with length prefix framing.
    int socket_send_blocking_lpf(int clientfd, uint8_t *buf, size_t size)
    {
        int ret;
        // hack: not very platform portable
        if ((ret = socket_send_blocking(clientfd, (uint8_t *)&size, sizeof(size))) < 0)
        {
            goto end;
        }

        if ((ret = socket_send_blocking(clientfd, buf, size)) < 0)
        {
            goto end;
        }
    end:
        return ret;
    }

    int socket_receive_blocking(int clientfd, uint8_t *buf, size_t size)
    {
        ssize_t ret;
        ssize_t recv = 0;

        while (recv < size)
        {
            ret = read(clientfd, buf + recv, size - recv);
            if (ret < 0)
            {
                // Buffer is full. Try again.
                if (errno == EAGAIN)
                {
                    continue;
                }
                // Misc error. Terminate the socket.
                tlog::error() << "socket_receive_blocking: " << std::string(std::strerror(errno));
                return -errno;
            }
            if (ret == 0 && recv < size)
            {
                // Client disconnected while sending data. Terminate the socket.
                tlog::error() << "socket_receive_blocking: Received EOF when transfer is not done.";
                return -1;
            }
            recv += ret;
        }

        return 0;
    }

    // Receive message with length prefix framing.
    std::string socket_receive_blocking_lpf(int clientfd)
    {
        int ret;
        size_t size;
        // hack: not very platform portable
        if ((ret = socket_receive_blocking(clientfd, (uint8_t *)&size, sizeof(size))) < 0)
        {
            throw std::runtime_error{"socket_receive_blocking_lpf: Error while receiving data size from socket."};
        }

        auto buffer = std::make_unique<char[]>(size);

        if ((ret = socket_receive_blocking(clientfd, (uint8_t *)buffer.get(), size)) < 0)
        {
            throw std::runtime_error{"socket_receive_blocking_lpf: Error while receiving data from socket."};
        }

        return std::string(buffer.get(), buffer.get() + size);
    }
}
