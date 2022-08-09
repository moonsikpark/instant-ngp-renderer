/*
 * Copyright (c) Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_main.h
 *  @author Moonsik Park, Korea Institute of Science and Technology
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/renderer_common.h>
#include <neural-graphics-primitives/testbed.h>

namespace nes
{
    void render_server(std::string &nes_addr, uint16_t nes_port, std::string &scene_location, std::string &snapshot_location, bool depth_test);
}
