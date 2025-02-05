/*
 * Copyright (c) 2020-2022, Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_common.cu
 *  @author Moonsik Park, Korean Institute of Science and Technology
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/renderer_common.h>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace nes
{

    Eigen::Matrix<float, 3, 4> cam_to_matrix(nesproto::Camera cam)
    {
        Eigen::Matrix<float, 3, 4> mat;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                mat(i, j) = cam.matrix(i * 4 + j);
            }
        }

        return mat;
    }

    void render(ngp::Testbed &testbed, nesproto::FrameRequest request, ngp::ERenderMode mode, float *fbuf, float *fbuf_depth)
    {
        uint32_t width = request.camera().width();
        uint32_t height = request.camera().height();
        testbed.m_render_mode = mode;
        testbed.m_windowless_render_surface.resize({width, height});
        testbed.m_windowless_render_surface_depth.resize({width, height});
        Eigen::Matrix<float, 3, 4> cam_matrix(cam_to_matrix(request.camera()));
        testbed.m_windowless_render_surface.reset_accumulation();
        testbed.m_windowless_render_surface_depth.reset_accumulation();
        testbed.render_frame_with_depth(cam_matrix, cam_matrix, Eigen::Vector4f::Zero(), testbed.m_windowless_render_surface, testbed.m_windowless_render_surface_depth, true);
        CUDA_CHECK_THROW(cudaMemcpy2DFromArray(fbuf, width * sizeof(float) * 4, testbed.m_windowless_render_surface.surface_provider().array(), 0, 0, width * sizeof(float) * 4, height, cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy2DFromArray(fbuf_depth, width * sizeof(float) * 4, testbed.m_windowless_render_surface_depth.surface_provider().array(), 0, 0, width * sizeof(float) * 4, height, cudaMemcpyDeviceToHost));

    }

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
        size_t size;
        // hack: not very platform portable
        if (socket_receive_blocking(clientfd, (uint8_t *)&size, sizeof(size)) < 0)
        {
            throw std::runtime_error{"socket_receive_blocking_lpf: Error while receiving data size from socket."};
        }

        auto buffer = std::make_unique<char[]>(size);

        if (socket_receive_blocking(clientfd, (uint8_t *)buffer.get(), size) < 0)
        {
            throw std::runtime_error{"socket_receive_blocking_lpf: Error while receiving data from socket."};
        }

        return std::string(buffer.get(), buffer.get() + size);
    }
}
