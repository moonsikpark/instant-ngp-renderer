/*
 * Copyright (c) Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_server.h
 *  @author Moonsik Park, Korea Institute of Science and Technology
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/testbed.h>

namespace nes
{
    void server_client_thread(int targetfd, ngp::Testbed &testbed, std::atomic<bool> &shutdown_requested, bool depth_test);
    void server_main_thread(std::string bind_addr, uint16_t bind_port, ngp::Testbed &testbed, std::atomic<bool> &shutdown_requested, bool depth_test);
}
