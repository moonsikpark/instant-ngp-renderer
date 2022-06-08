/*
 * Copyright (c) 2020-2022, Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_server.cu
 *  @author Moonsik Park, Korean Institute of Science and Technology
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/renderer_common.h>

#include <thread>

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arpa/inet.h>

namespace nes
{
    void server_client_thread(int targetfd, ngp::Testbed &testbed, std::atomic<bool> &shutdown_requested)
    {
        // todo: evaluate what size should this be
        int ret;
        size_t size = 1024 * 1024 * 50;
        auto fbuf = std::make_unique<float[]>(size);
        auto cbuf = std::make_unique<char[]>(size);

        while (!shutdown_requested)
        {
            nesproto::FrameRequest request;

            try
            {
                if (!request.ParseFromString(nes::socket_receive_blocking_lpf(targetfd)))
                {
                    tlog::error() << "server_client_thread: Failed to receive FrameRequest.";
                }
            }
            catch (const std::runtime_error &)
            {
                tlog::error() << "server_client_thread: Failed to receive request from client. Exiting.";
                break;
            }

            tlog::info() << "server_client_thread: Received FrameRequest index=" << request.index();

            nesproto::RenderedFrame frame;

            frame.set_index(request.index());

            frame.set_allocated_camera(new nesproto::Camera(request.camera()));

            // todo: don't blindly follow server's resolution direction
            frame.set_pixelformat(nesproto::RenderedFrame_PixelFormat::RenderedFrame_PixelFormat_BGR32);
            frame.set_frame(render(testbed, request, ngp::ERenderMode::Shade, fbuf.get(), cbuf.get()));
            // todo: one render can output frame and depth.
            // look at __global__ void shade_kernel_sdf()
            // frame.set_depth(render(testbed, request, ERenderMode::Depth, fbuf.get(), cbuf.get()));
            tlog::success() << "server_client_thread: Rendered frame index=" << request.index() << " width=" << request.camera().width() << " height=" << request.camera().height();

            std::string frame_buffer = frame.SerializeAsString();

            ret = nes::socket_send_blocking_lpf(targetfd, (uint8_t *)frame_buffer.data(), frame_buffer.size());
            if (ret < 0)
            {
                tlog::error() << "server_client_thread: Failed to send frame to client. Exiting.";
                break;
            }
            tlog::success() << "server_client_thread: Sent frame index=" << request.index();
        }
        close(targetfd);
        tlog::success() << "server_client_thread: Exiting thread.";
    }

    void server_main_thread(std::string bind_addr, uint16_t bind_port, ngp::Testbed &testbed, std::atomic<bool> &shutdown_requested)
    {
        tlog::info() << "server_main_thread: Initalizing server...";
        struct sockaddr_in addr;
        int sockfd, targetfd;

        if ((sockfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0)) < 0)
        {
            throw std::runtime_error{"server_main_thread: Failed to create socket: " + std::string(std::strerror(errno))};
        }

        // Bind the socket to the given address.
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(bind_addr.c_str());
        addr.sin_port = htons(bind_port);
        if ((bind(sockfd, (struct sockaddr *)&(addr), sizeof(addr))) < 0)
        {
            throw std::runtime_error{"server_main_thread: Failed to bind to socket: " + std::string(std::strerror(errno))};
        }

        // Listen to the socket.
        if ((listen(sockfd, /* backlog= */ 2)) < 0)
        {
            throw std::runtime_error{"server_main_thread: Failed to listen to socket: " + std::string(std::strerror(errno))};
        }

        tlog::success() << "server_main_thread: Socket server created and listening.";

        while (!shutdown_requested)
        {
            // Wait for connections, but don't block if there aren't any connections.
            if ((targetfd = accept4(sockfd, NULL, NULL, SOCK_NONBLOCK)) < 0)
            {
                // There are no clients to accept.
                if (errno == EAGAIN || errno == EINTR || errno == ECONNABORTED)
                {
                    // Sleep and wait again for connections.
                    tlog::info() << "server_main_thread: Waiting for ngp-encode-server to connect.";
                    std::this_thread::sleep_for(std::chrono::milliseconds{500});
                    continue;
                }
                else
                {
                    throw std::runtime_error{"server_main_thread: Failed to accept client: " + std::string(std::strerror(errno))};
                }
            }
            else
            {
                // A client wants to connect. Spawn a thread with the client's fd and process the client there.
                // TODO: maybe not thread per client but thread pool?
                std::thread _socket_client_thread(server_client_thread, targetfd, std::ref(testbed), std::ref(shutdown_requested));
                _socket_client_thread.detach();
                tlog::success() << "server_main_thread: Received client connection (targetfd=" << targetfd << ").";
            }
        }
        // Cleanup: close the socket.
        close(sockfd);
        tlog::success() << "server_main_thread: Exiting thread.";
    }
}
