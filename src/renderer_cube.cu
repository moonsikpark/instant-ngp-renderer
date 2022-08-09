/*
 * Copyright (c) 2020-2022, Moonsik Park.  All rights reserved.
 *
 */

/** @file   renderer_server.cu
 *  @author Moonsik Park, Korean Institute of Science and Technology
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/renderer_common.h>

#include <thread>
#include <cstddef>
#include <vector>
#include <atomic>

#include <GL/glut.h>

namespace nes
{
    void renderer_cube_thread(
        nesproto::FrameRequest &request,
        std::condition_variable &cv,
        std::mutex &mutex,
        bool &render_cube,
        float *buf,
        float *buf_depth,
        std::atomic<bool> &shutdown_requested)
    {
        int a = 1;
        glutInit(&a, NULL);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
        GLint w = 1300;
        GLint h = 1300;
        glutInitWindowSize(w, h);
        glutCreateWindow("Cyan Shapes in Yellow Light");

        // start init code
        GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
        GLfloat yellow[] = { 1.0, 1.0, 0.0, 1.0 };
        GLfloat cyan[] = { 0.0, 1.0, 1.0, 1.0 };
        GLfloat white[] = { 1.0, 1.0, 1.0, 1.0 };
        //GLfloat direction[] = { 1.0, 1.0, 1.0, 0.0 };
        GLfloat direction[] = { -1.0, -1.0, 1.0, 0.0 };

        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, cyan);
        glMaterialfv(GL_FRONT, GL_SPECULAR, white);
        glMaterialf(GL_FRONT, GL_SHININESS, 30);

        glLightfv(GL_LIGHT0, GL_AMBIENT, black);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, yellow);
        glLightfv(GL_LIGHT0, GL_SPECULAR, white);
        glLightfv(GL_LIGHT0, GL_POSITION, direction);

        glEnable(GL_LIGHTING);                // so the renderer considers light
        glEnable(GL_LIGHT0);                  // turn LIGHT0 on
        glEnable(GL_DEPTH_TEST);              // so the renderer considers depth
        // end init code

        float cam_matrix[16];
        cam_matrix[12] = 0;
        cam_matrix[13] = 0;
        cam_matrix[14] = 0;
        cam_matrix[15] = 1;

        int init = true;
        while (!shutdown_requested) {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [&]{return render_cube;});

            std::memcpy(&cam_matrix, (void *)request.camera().matrix().data(), sizeof(float) * 12);

            float x = cam_matrix[3];
            float y = cam_matrix[7];
            float z = cam_matrix[11];

            if (init || w != request.camera().width() || h != request.camera().height()) {
                if (init) {
                    init = false;
                } else {
                    w = request.camera().width();
                    h = request.camera().height();
                }
                tlog::info() << "resizing to w=" << w << " h=" << h;
                glViewport(0, 0, w, h);
                glMatrixMode(GL_PROJECTION);
                GLfloat aspect = GLfloat(w) / GLfloat(h);
                glLoadIdentity();
                if (w <= h) {
                    // width is smaller, so stretch out the height
                    //gluPerspective(-20, aspect, 1.f, 100.f);
                    gluPerspective(45, aspect, 1.0f, 100.f);
                }
                else {
                    // height is smaller, so stretch out the width
                    //gluPerspective(-20, aspect, 1.f, 100.f);
                    gluPerspective(45, aspect, 1.f, 100.f);
                }
            }

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glMatrixMode(GL_MODELVIEW);
            // Rotate the scene so we can see the tops of the shapes.

            // Make a torus floating 0.5 above the x-z plane.  The standard torus in
            // the GLUT library is, perhaps surprisingly, a stack of circles which
            // encircle the z-axis, so we need to rotate it 90 degrees about x to
            // get it the way we want.
            glPushMatrix();
            glTranslated(0, 1, -4);
            glRotatef(270, 1.0, 0.0, 0.0);
            glRotatef(-30, 0.0, 0.0, 1.0);
            glutSolidTorus(0.2, 0.8, 10, 20);
            glPopMatrix();

            glFlush();

            //std::memcpy(&cam_matrix, (void *)request.camera().matrix().data(), sizeof(float) * 12);

            //tlog::info() << "renderer_cube_thread: lock released, rendering opengl scene for scene #" << request.index();
            //tlog::info() << "renderer_cube_thread: window.width=" << window_width << " window_height=" << window_height;

            glReadPixels(0, 0, w, h, GL_RGB, GL_FLOAT, buf);
            glReadPixels(0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, buf_depth); 
            render_cube = false;
            cv.notify_one();
        }

    }

    __device__ uint8_t compare_pixels(float* __restrict__ depth_a, float* __restrict__ depth_b) {

    }

    __global__ void compareDepth() {
        
    }
}
