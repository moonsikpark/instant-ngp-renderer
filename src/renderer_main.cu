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

#include <neural-graphics-primitives/renderer_common.h>
#include <neural-graphics-primitives/renderer_main.h>
#include <neural-graphics-primitives/renderer_server.h>
#include <neural-graphics-primitives/renderer_cube.cuh>

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
    
    void render_server(std::string &nes_addr, uint16_t nes_port, std::string &scene_location, std::string &snapshot_location, bool depth_test)
    {
        signal(SIGINT, signal_handler);
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        Testbed testbed{ETestbedMode::Nerf};
        testbed.m_train = false;

        testbed.load_training_data(scene_location);
        testbed.load_snapshot(snapshot_location);

        std::thread _server_main_thread(server_main_thread, nes_addr, nes_port, std::ref(testbed), std::ref(shutdown_requested), depth_test);
        _server_main_thread.join();
        tlog::info() << "render_server: All threads exited. Exiting program.";
    }
}
