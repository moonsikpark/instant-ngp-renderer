/*
 * Copyright (c) Moonsik Park.  All rights reserved.
 *
 */

/** @file   nes_client.h
 *  @author Moonsik Park, Korea Institute of Science and Technology
 */

#pragma once

#include <proto/nes.pb.h>

namespace nes
{
    int socket_send_blocking(int clientfd, uint8_t *buf, size_t size);
    int socket_send_blocking_lpf(int clientfd, uint8_t *buf, size_t size);
    int socket_receive_blocking(int clientfd, uint8_t *buf, size_t size);
    std::string socket_receive_blocking_lpf(int clientfd);
}
