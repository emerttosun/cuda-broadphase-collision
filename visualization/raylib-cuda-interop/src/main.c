#include "RaylibInteropTypes.cuh"

#include "raylib.h"
#include "rlgl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kWindowWidth = 1280;
static const int kWindowHeight = 720;

static const char* kVertexShader =
    "#version 330 core\n"
    "layout(location = 0) in vec2 aPos;\n"
    "layout(location = 1) in vec2 aLocal;\n"
    "layout(location = 2) in vec3 aColor;\n"
    "out vec3 vColor;\n"
    "out vec2 vLocal;\n"
    "void main() {\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "    vColor = aColor;\n"
    "    vLocal = aLocal;\n"
    "}\n";

static const char* kFragmentShader =
    "#version 330 core\n"
    "in vec3 vColor;\n"
    "in vec2 vLocal;\n"
    "out vec4 FragColor;\n"
    "void main() {\n"
    "    float d = dot(vLocal, vLocal);\n"
    "    if (d > 1.0) discard;\n"
    "    float core = smoothstep(1.0, 0.0, d);\n"
    "    vec3 color = mix(vColor * 0.55, vColor, core);\n"
    "    FragColor = vec4(color, 1.0);\n"
    "}\n";

static void create_particle_buffers(unsigned int* vao, unsigned int* vbo, int object_count) {
    *vao = rlLoadVertexArray();
    rlEnableVertexArray(*vao);

    *vbo = rlLoadVertexBuffer(NULL, object_count * 6 * (int)sizeof(RenderVertex), true);
    rlEnableVertexBuffer(*vbo);

    rlSetVertexAttribute(0, 2, RL_FLOAT, false, sizeof(RenderVertex), 0);
    rlEnableVertexAttribute(0);
    rlSetVertexAttribute(1, 2, RL_FLOAT, false, sizeof(RenderVertex), 2 * sizeof(float));
    rlEnableVertexAttribute(1);
    rlSetVertexAttribute(2, 3, RL_FLOAT, false, sizeof(RenderVertex), 4 * sizeof(float));
    rlEnableVertexAttribute(2);

    rlDisableVertexBuffer();
    rlDisableVertexArray();
}

static void draw_particles(unsigned int vao, unsigned int shader_program, int object_count) {
    rlDrawRenderBatchActive();
    rlEnableShader(shader_program);
    if (rlEnableVertexArray(vao)) {
        rlDrawVertexArray(0, object_count * 6);
        rlDisableVertexArray();
    }
    rlDisableShader();
}

static const int kCountStep = 500;
static const int kCountMin = 100;
static const int kCountMax = 20000;

static const char* mode_name(int mode) {
    switch (mode) {
        case VISUALIZER_MODE_CUDA_UNIFORM_GRID: return "cuda_uniform_grid";
        case VISUALIZER_MODE_CPU_BRUTE_FORCE:   return "cpu_brute_force";
        default:                                return "cuda_brute_force";
    }
}

static int recreate_simulation(unsigned int* vao, unsigned int* vbo,
                               int new_count, int mode, int clustered) {
    cuda_visualizer_destroy();
    rlUnloadVertexBuffer(*vbo);
    rlUnloadVertexArray(*vao);
    create_particle_buffers(vao, vbo, new_count);
    if (!cuda_visualizer_create(*vbo, new_count, kWindowWidth, kWindowHeight)) {
        return 0;
    }
    cuda_visualizer_set_mode(mode);
    cuda_visualizer_reset(clustered);
    return 1;
}

int main(int argc, char** argv) {
    int object_count = 2500;
    if (argc > 1) {
        object_count = atoi(argv[1]);
        if (object_count < 100) {
            object_count = 100;
        }
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(kWindowWidth, kWindowHeight, "Raylib + CUDA OpenGL Interop Collision Visualizer");
    SetTargetFPS(60);
    unsigned int particle_shader = rlLoadShaderProgram(kVertexShader, kFragmentShader);
    if (particle_shader == 0) {
        fprintf(stderr, "Failed to create particle shader.\n");
        return 1;
    }

    unsigned int vao = 0;
    unsigned int vbo = 0;
    create_particle_buffers(&vao, &vbo, object_count);

    if (!cuda_visualizer_create(vbo, object_count, kWindowWidth, kWindowHeight)) {
        fprintf(stderr, "Failed to create CUDA visualizer.\n");
        return 1;
    }

    int clustered = 0;
    int paused = 0;
    int mode = VISUALIZER_MODE_CUDA_BRUTE_FORCE;
    VisualizerMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_C)) {
            clustered = !clustered;
            cuda_visualizer_reset(clustered);
        }
        if (IsKeyPressed(KEY_R)) {
            cuda_visualizer_reset(clustered);
        }
        if (IsKeyPressed(KEY_G)) {
            mode = (mode + 1) % 3;
            cuda_visualizer_set_mode(mode);
        }

        int new_count = object_count;
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
            new_count = object_count + kCountStep;
            if (new_count > kCountMax) new_count = kCountMax;
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
            new_count = object_count - kCountStep;
            if (new_count < kCountMin) new_count = kCountMin;
        }
        if (new_count != object_count) {
            if (recreate_simulation(&vao, &vbo, new_count, mode, clustered)) {
                object_count = new_count;
                memset(&metrics, 0, sizeof(metrics));
            }
        }

        if (!paused) {
            cuda_visualizer_step(GetFrameTime(), &metrics);
        }

        BeginDrawing();
        ClearBackground((Color){8, 10, 14, 255});
        draw_particles(vao, particle_shader, object_count);

        const char* compute_label =
            (mode == VISUALIZER_MODE_CPU_BRUTE_FORCE) ? "CPU compute" : "CUDA compute";

        DrawRectangle(12, 12, 480, 220, (Color){18, 22, 30, 220});
        DrawText("Raylib + CUDA-OpenGL Interop", 24, 24, 20, RAYWHITE);
        DrawText(TextFormat("Objects: %d  [+/-]", object_count), 24, 52, 18, LIGHTGRAY);
        DrawText(TextFormat("Distribution: %s  [C]", clustered ? "clustered" : "uniform"), 24, 76, 18, LIGHTGRAY);
        DrawText(TextFormat("Method: %s  [G]", mode_name(mode)), 24, 100, 18, LIGHTGRAY);
        DrawText(TextFormat("Collisions: %llu", metrics.collision_count), 24, 124, 18, LIGHTGRAY);
        DrawText(TextFormat("Candidate pairs: %llu", metrics.candidate_pair_count), 24, 148, 18, LIGHTGRAY);
        DrawText(TextFormat("%s: %.3f ms", compute_label, (double)metrics.gpu_time_ms), 24, 172, 18, LIGHTGRAY);
        DrawText(TextFormat("FPS: %d", GetFPS()), 24, 196, 18, LIGHTGRAY);

        EndDrawing();
    }

    cuda_visualizer_destroy();
    rlUnloadVertexBuffer(vbo);
    rlUnloadVertexArray(vao);
    rlUnloadShaderProgram(particle_shader);
    CloseWindow();
    return 0;
}
