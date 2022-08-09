/*
 * Copyright (c) Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_cube.cuh
 *  @author Moonsik Park, Korea Institute of Science and Technology
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/testbed.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace nes
{
    GLuint LoadShaders(std::string vertex_shader_code, std::string fragment_shader_code);
    void renderer_cube_thread(nesproto::FrameRequest &request, std::condition_variable &cv, std::mutex &mutex, bool &render_cube, float *buf, float *buf_depth, std::atomic<bool> &shutdown_requested);
}
