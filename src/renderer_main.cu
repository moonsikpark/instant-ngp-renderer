/*
 * Copyright (c) 2020-2022, Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_main.cu
 *  @author Moonsik Park, Korean Institute of Science and Technology
 */

#include <csignal>
#include <thread>

#include <neural-graphics-primitives/testbed.h>

#include <neural-graphics-primitives/renderer_main.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>
#include <filesystem/path.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arpa/inet.h>

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;
using namespace Eigen;
namespace fs = filesystem;

namespace nes
{
    std::atomic<bool> shutdown_requested{false};

    void signal_handler(int)
    {
        shutdown_requested = true;
    }

    Matrix<float, 3, 4> cam_to_matrix(nesproto::Camera cam)
    {
        Matrix<float, 3, 4> mat;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                mat(i, j) = cam.matrix(i * 4 + j);
            }
        }

        return mat;
    }

    std::string render(Testbed &testbed, nesproto::FrameRequest request, ERenderMode mode, float *fbuf, char *cbuf)
    {
        testbed.m_render_mode = mode;
        testbed.m_windowless_render_surface.resize({request.width(), request.height()});
        Matrix<float, 3, 4> cam_matrix(cam_to_matrix(request.camera()));
        tlog::info() << "Received FrameRequest index=" << request.index();
        testbed.m_windowless_render_surface.reset_accumulation();
        testbed.render_frame(cam_matrix, cam_matrix, Vector4f::Zero(), testbed.m_windowless_render_surface, true);

        CUDA_CHECK_THROW(cudaMemcpy2DFromArray(fbuf, request.width() * sizeof(float) * 4, testbed.m_windowless_render_surface.surface_provider().array(), 0, 0, request.width() * sizeof(float) * 4, request.height(), cudaMemcpyDeviceToHost));

        size_t size;

        if (mode == ERenderMode::Shade)
        {
            size = request.width() * request.height() * 4;
        }
        else
        {
            size = request.width() * request.height();
        }

        for (int i = 0; i < size; i++)
        {
            cbuf[i] = static_cast<int>(fbuf[i] * 255);
        }

        return std::string(cbuf, cbuf + size);
    }

    int nes_connect(std::string &nes_addr, uint16_t nes_port)
    {
        int fd;

        struct sockaddr_in addr;

        tlog::info() << "nes_connect: Connecting to server at " << nes_addr;

        if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        {
            throw std::runtime_error{"nes_connect: Failed to create socket: " + std::string(std::strerror(errno))};
        }

        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(nes_addr.c_str());
        addr.sin_port = htons(nes_port);
        if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        {
            throw std::runtime_error{"nes_connect: Failed to bind to socket: " + std::string(std::strerror(errno))};
        }

        // Listen to the socket.
        if ((listen(fd, /* backlog= */ 2)) < 0)
        {
            throw std::runtime_error{"Failed to listen to socket: " + std::string(std::strerror(errno))};
        }
        // Todo: make socket unblocking.
        if ((fd = accept(fd, NULL, NULL)) < 0)
        {
            throw std::runtime_error{"socket_main_thread: Failed to accept client: " + std::string(std::strerror(errno))};
        }

        return fd;
    }

    void render_server(std::string &nes_addr, uint16_t nes_port, std::string &scene_location, std::string &snapshot_location)
    {
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        signal(SIGINT, signal_handler);

        int serverfd = nes_connect(nes_addr, nes_port);
        size_t size = 1024 * 1024 * 50;
        auto fbuf = std::make_unique<float[]>(size);
        auto cbuf = std::make_unique<char[]>(size);

        Testbed testbed{ETestbedMode::Nerf};
        testbed.load_training_data(scene_location);
        testbed.load_snapshot(snapshot_location);

        // Disable training.
        testbed.m_train = false;

        while (!shutdown_requested)
        {
            nesproto::FrameRequest request;

            std::string a = nes::socket_receive_blocking_lpf(serverfd);

            if (!request.ParseFromString(a))
            {
                tlog::error() << "Failed to receive FrameRequest.";
            }
            tlog::info() << "Received FrameRequest index=" << request.index();

            nesproto::RenderedFrame frame;

            frame.set_index(request.index());
            frame.set_allocated_camera(new nesproto::Camera(request.camera()));

            // todo: don't blindly follow server's resolution direction
            frame.set_width(request.width());
            frame.set_height(request.height());
            frame.set_pixelformat(nesproto::RenderedFrame_PixelFormat::RenderedFrame_PixelFormat_BGR32);
            // todo: one render can output frame and depth.
            // look at __global__ void shade_kernel_sdf()
            frame.set_frame(render(testbed, request, ERenderMode::Shade, fbuf.get(), cbuf.get()));
            frame.set_depth(render(testbed, request, ERenderMode::Depth, fbuf.get(), cbuf.get()));

            std::string frame_serialized = frame.SerializeAsString();

            nes::socket_send_blocking_lpf(serverfd, (uint8_t *)frame_serialized.data(), frame_serialized.size());

            tlog::info() << "Sent RenderedFrame index=" << request.index();
        }
    }
}
