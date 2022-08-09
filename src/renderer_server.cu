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
#include <neural-graphics-primitives/renderer_cube.cuh>

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
    
    void server_client_thread(int targetfd, ngp::Testbed &testbed, std::atomic<bool> &shutdown_requested, bool depth_test)
    {
        // todo: evaluate what size should this be
        int ret;
        size_t size = 2000*2000*4;
        auto fbuf = std::make_unique<float[]>(size);
        auto fbuf_depth = std::make_unique<float[]>(size);
        auto fbuf_cube = std::make_unique<float[]>(size);
        auto fbuf_cube_depth = std::make_unique<float[]>(size);
        auto cbuf_depth_tested = std::make_unique<char[]>(size);

        auto cam_matrix = std::make_unique<float[]>(16);
        nesproto::FrameRequest request;


        std::condition_variable cv;
        std::mutex mutex;
        bool render_cube = false;
        std::thread _renderer_cube_thread;

        if (depth_test) {
            _renderer_cube_thread = std::thread(
                renderer_cube_thread,
                std::ref(request),
                std::ref(cv),
                std::ref(mutex),
                std::ref(render_cube),
                fbuf_cube.get(),
                fbuf_cube_depth.get(),
                std::ref(shutdown_requested)
            );

        }

        while (!shutdown_requested)
        {
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

            frame.mutable_camera()->CopyFrom(request.camera());


            int idx = 0;
            if (depth_test) {
                render_cube = true;
                cv.notify_one();
                render(testbed, request, ngp::ERenderMode::Shade, fbuf.get(), fbuf_depth.get());
                std::unique_lock<std::mutex> lk(mutex);
                // Render NeRF while rendering OpenGL. 
                cv.wait(lk, [&]{return !render_cube;});

                
                for(int i = 0; i < request.camera().width() * request.camera().height(); i++) {
                    if (fbuf_cube_depth[i] < 0.9) {
                        fbuf_cube_depth[i] -= 0.5;
                    }

                    if (fbuf_cube_depth[i] < (fbuf_depth[i*4] * 0.03125f)) {
                        cbuf_depth_tested[idx] = static_cast<int>(fbuf_cube[i*3] * 255);
                        cbuf_depth_tested[idx+1] = static_cast<int>(fbuf_cube[i*3+1] * 255);
                        cbuf_depth_tested[idx+2] = static_cast<int>(fbuf_cube[i*3+2] * 255);
                    } else {
                        cbuf_depth_tested[idx] = static_cast<int>(fbuf[i*4] * 255);
                        cbuf_depth_tested[idx+1] = static_cast<int>(fbuf[i*4+1] * 255);
                        cbuf_depth_tested[idx+2] = static_cast<int>(fbuf[i*4+2] * 255);
                    }
                    idx += 3;
                }
            } else {
                for(int i = 0; i < request.camera().width() * request.camera().height(); i++) {
                        cbuf_depth_tested[idx] = static_cast<int>(fbuf[i*4] * 255);
                        cbuf_depth_tested[idx+1] = static_cast<int>(fbuf[i*4+1] * 255);
                        cbuf_depth_tested[idx+2] = static_cast<int>(fbuf[i*4+2] * 255);
                    idx += 3;
                }
            }


            std::string depth_tested_frame(cbuf_depth_tested.get(), cbuf_depth_tested.get() + request.camera().width() * request.camera().height() * 3);

            frame.set_frame(depth_tested_frame);
            frame.set_depth(depth_tested_frame);
            frame.set_is_left(request.is_left());
            
            std::string direction = request.is_left() ? "left" : "right";
            tlog::success() << "server_client_thread: Rendered " << direction << " frame index=" << request.index() << " width=" << request.camera().width() << " height=" << request.camera().height();

            std::string frame_buffer = frame.SerializeAsString();

            ret = nes::socket_send_blocking_lpf(targetfd, (uint8_t *)frame_buffer.data(), frame_buffer.size());
            if (ret < 0)
            {
                tlog::error() << "server_client_thread: Failed to send frame to client. Exiting.";
                break;
            }
            tlog::success() << "server_client_thread: Sent frame index=" << request.index();
        }
        if (depth_test) {
            _renderer_cube_thread.join();
        }
        close(targetfd);
        tlog::success() << "server_client_thread: Exiting thread.";
    }

    void server_main_thread(std::string bind_addr, uint16_t bind_port, ngp::Testbed &testbed, std::atomic<bool> &shutdown_requested, bool depth_test)
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
                    tlog::info() << "server_main_thread: Waiting for ngp-encode-server to connect at " << bind_addr << ":" << bind_port;
                    std::this_thread::sleep_for(std::chrono::milliseconds{1000});
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
                std::thread _socket_client_thread(server_client_thread, targetfd, std::ref(testbed), std::ref(shutdown_requested), depth_test);
                _socket_client_thread.detach();
                tlog::success() << "server_main_thread: Received client connection (targetfd=" << targetfd << ").";
            }
        }
        // Cleanup: close the socket.
        close(sockfd);
        tlog::success() << "server_main_thread: Exiting thread.";
    }
}
