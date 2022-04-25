/*
 * Copyright (c) Moonsik Park.  All rights reserved.
 *
 */

/** @file   nes_client.h
 *  @author Moonsik Park, Korea Institute of Science and Technology
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nes_common.h>
#include <neural-graphics-primitives/testbed.h>

namespace nes
{
    void nes_client(std::string &nes_addr, uint16_t nes_port, std::string &scene_location, std::string &snapshot_location);
}
