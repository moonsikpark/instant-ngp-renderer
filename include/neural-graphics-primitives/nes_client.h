/*
 * Copyright (c) Moonsik Park.  All rights reserved.
 *
 */

/** @file   nesclient.h
 *  @author Moonsik Park, Korean Institute of Science and Technology
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/testbed.h>

typedef struct
{
    uint32_t width;
    uint32_t height;
    float rotx;
    float roty;
    float dx;
    float dy;
    float dz;
} __attribute__((packed)) Request;

typedef struct
{
    uint32_t filesize;
} __attribute__((packed)) RequestResponse;

namespace nes
{
    void nes_client(std::string &nes_addr);
}
