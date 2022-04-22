/*
 * Copyright (c) 2020-2022, Moonsik Park.  All rights reserved.
 *
 */

/** @file   nes_client.cu
 *  @author Moonsik Park, Korean Institute of Science and Technology
 */

#include <csignal>

#include <neural-graphics-primitives/testbed.h>

#include <neural-graphics-primitives/nes_client.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>
#include <filesystem/path.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;
using namespace Eigen;
namespace fs = filesystem;

/*
TODO: Architecture

1. Connect to the server.
2. Receive a request struct and decode it.
3. Generate response.
4. Send size and response.
5. Go to 1 until the server disconnects.
6. Exit the program.

TODO:
gracefully handle kill signals.

*/

namespace nes
{

    volatile std::sig_atomic_t signal_status;

    void signal_handler(int signum)
    {
        tlog::info() << "Ctrl+C received. Quitting.";
        signal_status = signum;
    }

    void render(Testbed &testbed, uint32_t width, uint32_t height, float *fbuf, float rotx, float roty, float dx, float dy, float dz)
    {
        tlog::info() << "rotx= " << rotx << " roty=" << roty;
        // get camera pov
        // Matrix<float, 3, 4> cam_matrix = testbed.m_smoothed_camera;
        Matrix<float, 3, 4> cam_matrix = Eigen::Matrix<float, 3, 4>::Zero();

        cam_matrix << 1.0f, 0.0f, 0.0f, 0.5f,
            0.0f, -1.0f, 0.0f, 0.5f,
            0.0f, 0.0f, -1.0f, 0.5f;

        // Angle movement
        Vector3f rel2 = {rotx, roty, 0.0f};
        // Vector3f rel2 = {dx, dy, dz};
        cam_matrix.col(3) += cam_matrix.block<3, 3>(0, 0) * rel2 * testbed.m_bounding_radius;

        // init pov translation vector
        Vector3f translate_vec = Vector3f::Zero();
        // x, y, z movement
        translate_vec.x() += dx;
        translate_vec.y() += dy;
        translate_vec.z() += dz;
        cam_matrix.col(3) += cam_matrix.block<3, 3>(0, 0) * translate_vec * testbed.m_bounding_radius;

        testbed.m_windowless_render_surface.reset_accumulation();
        tlog::info() << "reset_accumulation testbed";

        // render frame
        testbed.render_frame(cam_matrix, cam_matrix, Vector4f::Zero(), testbed.m_windowless_render_surface, true);

        // copy the render surface to a buffer
        CUDA_CHECK_THROW(cudaMemcpy2DFromArray(fbuf, width * sizeof(float) * 4, testbed.m_windowless_render_surface.surface_provider().array(), 0, 0, width * sizeof(float) * 4, height, cudaMemcpyDeviceToHost));
    }

    void nes_client(std::string &nes_addr)
    {
        signal(SIGINT, signal_handler);
        const char *address = nes_addr.c_str();
        uint32_t width = 1280;
        uint32_t height = 720;
        struct sockaddr_un addr;
        int fd;

        tlog::info() << "resized testbed";
        // reset accumulation to draw a new scene
        tlog::info() << "render_frame testbed";

        tlog::info() << "rendered";
        tlog::info() << "Connecting to server at " << nes_addr;

        if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
        {
            perror("socket error");
            exit(-1);
        }

        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        if (*address == '\0')
        {
            *addr.sun_path = '\0';
            strncpy(addr.sun_path + 1, address + 1, sizeof(addr.sun_path) - 2);
        }
        else
        {
            strncpy(addr.sun_path, address, sizeof(addr.sun_path) - 1);
        }

        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == -1)
        {
            throw std::runtime_error{"Failed to connect: " + std::string(std::strerror(errno))};
        }

        // create buffer that stores the float image from the neural network
        float *fbuf = (float *)malloc(sizeof(float) * width * height * 4);
        uint8_t *buf = (uint8_t *)malloc(sizeof(uint8_t) * width * height * 4);

        uint32_t filesize = width * height * 4 * sizeof(uint8_t);
        int ret, rc;

        Testbed testbed{ETestbedMode::Nerf};
        testbed.load_training_data("/home/test/instant-ngp/data/video/livinglab2-center-trashcan-chair");
        testbed.load_snapshot("/home/test/instant-ngp/livinglab2_100000.msgpack");
        tlog::info() << "created testbed";
        testbed.m_train = false;

        testbed.m_windowless_render_surface.resize({width, height});

        tlog::info() << "connected";
        while (1)
        {
            if (signal_status)
            {
                break;
            }
            Request req;
            RequestResponse resp;

            ret = (rc = read(fd, &req, sizeof(Request)));
            if (ret < 0)
            {
                throw std::runtime_error{"Failed to receive Request: " + std::string(std::strerror(errno))};
            }

            tlog::info() << "Received Request: width=" << req.width << " height=" << req.height << " rotx=" << req.rotx << " roty=" << req.roty << " dx=" << req.dx << " dy=" << req.dy << " dz=" << req.dz;

            render(testbed, req.width, req.height, fbuf, req.rotx, req.roty, req.dx, req.dy, req.dz);
            // create buffer that stores the converted BGRA uint8 image

            // normalize float image with range range 0...1 to uint8_t with range 0...256
            // TODO: check the magnitude of lost accuracy
            // TODO: check whether there are better ways (algorithm, parallel etc.) to do this
            for (int i = 0; i < width * height * 4; i++)
            {
                buf[i] = static_cast<int>(fbuf[i] * 255);
            }
            resp.filesize = filesize;
            ret = (rc = write(fd, &resp, sizeof(RequestResponse)));

            if (ret < 0)
            {
                throw std::runtime_error{"Failed to send RequestResponse: " + std::string(std::strerror(errno))};
            }

            tlog::info() << "Sent RequestResponse: filesize=" << resp.filesize;

            uint32_t sent = 0;
            auto progress = tlog::progress(filesize);
            while (sent < filesize)
            {
                ret = (rc = write(fd, buf + sent, filesize - sent));
                if (ret < 0)
                {
                    throw std::runtime_error{"Failed while sending image: " + std::string(std::strerror(errno))};
                }
                else
                {
                    sent += rc;
                    progress.update(sent);
                }
            }

            tlog::success() << "Successfully sent image after " << tlog::durationToString(progress.duration());
        }

        tlog::info() << "sent all";
    }
}
