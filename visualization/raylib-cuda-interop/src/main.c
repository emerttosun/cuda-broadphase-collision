#include "RaylibInteropTypes.cuh"

#include "raylib.h"
#include "rlgl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kWindowWidth = 1280;
static const int kWindowHeight = 720;
static const float kBoxDepth = 540.0f;
static const float kProjectionSkewX = 0.38f;
static const float kProjectionSkewY = 0.22f;
static const float kCameraDistance = 900.0f;
static const float kMouseColliderRadius = 54.0f;

static const char* kVertexShader =
    "#version 330 core\n"
    "layout(location = 0) in vec2 aPos;\n"
    "layout(location = 1) in float aDepth;\n"
    "layout(location = 2) in vec2 aLocal;\n"
    "layout(location = 3) in vec3 aColor;\n"
    "out vec3 vColor;\n"
    "out vec2 vLocal;\n"
    "out float vDepth;\n"
    "void main() {\n"
    "    gl_Position = vec4(aPos, aDepth, 1.0);\n"
    "    vColor = aColor;\n"
    "    vLocal = aLocal;\n"
    "    vDepth = aDepth;\n"
    "}\n";

static const char* kFragmentShader =
    "#version 330 core\n"
    "in vec3 vColor;\n"
    "in vec2 vLocal;\n"
    "in float vDepth;\n"
    "out vec4 FragColor;\n"
    "void main() {\n"
    "    float d = dot(vLocal, vLocal);\n"
    "    if (d > 1.0) discard;\n"
    "    float core = smoothstep(1.0, 0.0, d);\n"
    "    vec3 color = mix(vColor * 0.45, vColor, core);\n"
    "    color *= mix(0.68, 1.08, clamp((vDepth + 1.0) * 0.5, 0.0, 1.0));\n"
    "    FragColor = vec4(color, 1.0);\n"
    "}\n";

static void create_particle_buffers(unsigned int* vao, unsigned int* vbo, int object_count) {
    *vao = rlLoadVertexArray();
    rlEnableVertexArray(*vao);

    *vbo = rlLoadVertexBuffer(NULL, object_count * 6 * (int)sizeof(RenderVertex), true);
    rlEnableVertexBuffer(*vbo);

    rlSetVertexAttribute(0, 2, RL_FLOAT, false, sizeof(RenderVertex), 0);
    rlEnableVertexAttribute(0);
    rlSetVertexAttribute(1, 1, RL_FLOAT, false, sizeof(RenderVertex), 2 * sizeof(float));
    rlEnableVertexAttribute(1);
    rlSetVertexAttribute(2, 2, RL_FLOAT, false, sizeof(RenderVertex), 3 * sizeof(float));
    rlEnableVertexAttribute(2);
    rlSetVertexAttribute(3, 3, RL_FLOAT, false, sizeof(RenderVertex), 5 * sizeof(float));
    rlEnableVertexAttribute(3);

    rlDisableVertexBuffer();
    rlDisableVertexArray();
}

static Vector2 project_box_point(float x, float y, float z) {
    const float depth_centered = z - kBoxDepth * 0.5f;
    const float perspective = kCameraDistance / (kCameraDistance + (kBoxDepth - z));
    const float screen_x =
        ((x - kWindowWidth * 0.5f) + depth_centered * kProjectionSkewX) * perspective
        + kWindowWidth * 0.5f;
    const float screen_y =
        ((y - kWindowHeight * 0.5f) - depth_centered * kProjectionSkewY) * perspective
        + kWindowHeight * 0.5f;
    return (Vector2){screen_x, screen_y};
}

static void draw_container_box(int draw_faces) {
    const float left = 0.0f;
    const float right = (float)kWindowWidth;
    const float top = 0.0f;
    const float bottom = (float)kWindowHeight;
    const float z_near = kBoxDepth;
    const float z_far = 0.0f;
    const Color back = (Color){86, 111, 138, 90};
    const Color front = (Color){150, 206, 255, 150};

    Vector2 p000 = project_box_point(left, top, z_far);
    Vector2 p100 = project_box_point(right, top, z_far);
    Vector2 p110 = project_box_point(right, bottom, z_far);
    Vector2 p010 = project_box_point(left, bottom, z_far);
    Vector2 p001 = project_box_point(left, top, z_near);
    Vector2 p101 = project_box_point(right, top, z_near);
    Vector2 p111 = project_box_point(right, bottom, z_near);
    Vector2 p011 = project_box_point(left, bottom, z_near);

    if (draw_faces) {
        DrawTriangle(p000, p100, p110, (Color){32, 56, 78, 26});
        DrawTriangle(p000, p110, p010, (Color){32, 56, 78, 26});
        DrawTriangle(p001, p111, p101, (Color){76, 130, 176, 18});
        DrawTriangle(p001, p011, p111, (Color){76, 130, 176, 18});
    }

    DrawLineV(p000, p100, back);
    DrawLineV(p100, p110, back);
    DrawLineV(p110, p010, back);
    DrawLineV(p010, p000, back);
    DrawLineV(p001, p101, front);
    DrawLineV(p101, p111, front);
    DrawLineV(p111, p011, front);
    DrawLineV(p011, p001, front);
    DrawLineV(p000, p001, (Color){122, 170, 210, 120});
    DrawLineV(p100, p101, (Color){122, 170, 210, 120});
    DrawLineV(p110, p111, (Color){122, 170, 210, 120});
    DrawLineV(p010, p011, (Color){122, 170, 210, 120});
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
        case VISUALIZER_MODE_CUDA_LBVH:         return "cuda_lbvh";
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
            mode = (mode + 1) % 4;
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

        Vector2 mouse = GetMousePosition();
        const int mouse_active =
            mouse.x >= 0.0f && mouse.x < (float)kWindowWidth
            && mouse.y >= 0.0f && mouse.y < (float)kWindowHeight;

        if (!paused) {
            cuda_visualizer_step(GetFrameTime(), mouse.x, mouse.y, mouse_active, &metrics);
        }

        BeginDrawing();
        ClearBackground((Color){8, 10, 14, 255});
        draw_container_box(1);
        draw_particles(vao, particle_shader, object_count);
        draw_container_box(0);
        if (mouse_active) {
            DrawCircleLines((int)mouse.x, (int)mouse.y, kMouseColliderRadius, (Color){255, 255, 255, 95});
            DrawCircle((int)mouse.x, (int)mouse.y, 3.0f, (Color){255, 255, 255, 160});
        }

        const char* compute_label =
            (mode == VISUALIZER_MODE_CPU_BRUTE_FORCE) ? "CPU compute" : "CUDA compute";

        DrawRectangle(12, 12, 480, 220, (Color){18, 22, 30, 220});
        DrawText("Raylib + CUDA-OpenGL Interop 3D", 24, 24, 20, RAYWHITE);
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
