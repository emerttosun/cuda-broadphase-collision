#include "RaylibInteropTypes.cuh"

#include "raylib.h"
#include "rlgl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kWindowWidth = 1280;
static const int kWindowHeight = 720;
static const float kBoxDepth = 540.0f;
static const float kMouseColliderRadius = 54.0f;
static const float kCameraMinDistance = 90.0f;
static const float kCameraMaxDistance = 2200.0f;

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

static float clamp_float(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static Vector2 project_box_point(float x, float y, float z, const VisualizerCamera* camera) {
    const float cy = cosf(camera->yaw);
    const float sy = sinf(camera->yaw);
    const float cp = cosf(camera->pitch);
    const float sp = sinf(camera->pitch);

    const float forward_x = sy * cp;
    const float forward_y = sp;
    const float forward_z = cy * cp;
    const float right_x = cy;
    const float right_z = -sy;
    const float up_x = -sy * sp;
    const float up_y = cp;
    const float up_z = -cy * sp;

    const float world_x = x - kWindowWidth * 0.5f;
    const float world_y = kWindowHeight * 0.5f - y;
    const float world_z = z - kBoxDepth * 0.5f;
    const float rel_x = world_x + forward_x * camera->distance;
    const float rel_y = world_y + forward_y * camera->distance;
    const float rel_z = world_z + forward_z * camera->distance;

    const float camera_x = rel_x * right_x + rel_z * right_z;
    const float camera_y = rel_x * up_x + rel_y * up_y + rel_z * up_z;
    float camera_z = rel_x * forward_x + rel_y * forward_y + rel_z * forward_z;
    if (camera_z < 20.0f) {
        camera_z = 20.0f;
    }

    const float focal = 760.0f;
    return (Vector2){
        kWindowWidth * 0.5f + focal * camera_x / camera_z,
        kWindowHeight * 0.5f - focal * camera_y / camera_z
    };
}

static void draw_container_box(int draw_faces, const VisualizerCamera* camera) {
    const float left = 0.0f;
    const float right = (float)kWindowWidth;
    const float top = 0.0f;
    const float bottom = (float)kWindowHeight;
    const float z_near = kBoxDepth;
    const float z_far = 0.0f;
    const Color back = (Color){86, 111, 138, 90};
    const Color front = (Color){150, 206, 255, 150};

    Vector2 p000 = project_box_point(left, top, z_far, camera);
    Vector2 p100 = project_box_point(right, top, z_far, camera);
    Vector2 p110 = project_box_point(right, bottom, z_far, camera);
    Vector2 p010 = project_box_point(left, bottom, z_far, camera);
    Vector2 p001 = project_box_point(left, top, z_near, camera);
    Vector2 p101 = project_box_point(right, top, z_near, camera);
    Vector2 p111 = project_box_point(right, bottom, z_near, camera);
    Vector2 p011 = project_box_point(left, bottom, z_near, camera);

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

static const char* visualizer_radius_profile_name(RadiusProfile profile) {
    return radius_profile_name(profile);
}

static const char* mode_name(int mode) {
    switch (mode) {
        case VISUALIZER_MODE_CUDA_UNIFORM_GRID: return "cuda_uniform_grid";
        case VISUALIZER_MODE_CPU_BRUTE_FORCE:   return "cpu_brute_force";
        case VISUALIZER_MODE_CUDA_LBVH:         return "cuda_lbvh";
        case VISUALIZER_MODE_CUDA_UNIFORM_GRID_3D: return "cuda_uniform_grid_3d";
        default:                                return "cuda_brute_force";
    }
}

static int recreate_simulation(unsigned int* vao, unsigned int* vbo,
                               int new_count, int mode, int clustered, RadiusProfile radius_profile) {
    cuda_visualizer_destroy();
    rlUnloadVertexBuffer(*vbo);
    rlUnloadVertexArray(*vao);
    create_particle_buffers(vao, vbo, new_count);
    if (!cuda_visualizer_create(*vbo, new_count, kWindowWidth, kWindowHeight)) {
        return 0;
    }
    cuda_visualizer_set_mode(mode);
    cuda_visualizer_reset(clustered, radius_profile);
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
    RadiusProfile radius_profile = RADIUS_PROFILE_NARROW;
    VisualizerCamera camera = {0.0f, 0.0f, 900.0f};
    VisualizerMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    while (!WindowShouldClose()) {
        const float frame_dt = GetFrameTime();
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_C)) {
            clustered = !clustered;
            cuda_visualizer_reset(clustered, radius_profile);
            memset(&metrics, 0, sizeof(metrics));
        }
        if (IsKeyPressed(KEY_V)) {
            radius_profile = radius_profile_next(radius_profile);
            cuda_visualizer_reset(clustered, radius_profile);
            memset(&metrics, 0, sizeof(metrics));
        }
        if (IsKeyPressed(KEY_R)) {
            cuda_visualizer_reset(clustered, radius_profile);
            memset(&metrics, 0, sizeof(metrics));
        }
        if (IsKeyPressed(KEY_G)) {
            mode = (mode + 1) % 5;
            cuda_visualizer_set_mode(mode);
        }
        if (IsKeyPressed(KEY_F)) {
            camera.yaw = 0.0f;
            camera.pitch = 0.0f;
            camera.distance = 900.0f;
        }

        const float turn_speed = 1.45f * frame_dt;
        const float move_speed = 650.0f * frame_dt;
        if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) camera.yaw -= turn_speed;
        if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) camera.yaw += turn_speed;
        if (IsKeyDown(KEY_Q) || IsKeyDown(KEY_UP)) camera.pitch += turn_speed;
        if (IsKeyDown(KEY_E) || IsKeyDown(KEY_DOWN)) camera.pitch -= turn_speed;
        if (IsKeyDown(KEY_W)) camera.distance -= move_speed;
        if (IsKeyDown(KEY_S)) camera.distance += move_speed;
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            const Vector2 delta = GetMouseDelta();
            camera.yaw += delta.x * 0.006f;
            camera.pitch -= delta.y * 0.006f;
        }
        camera.distance -= GetMouseWheelMove() * 90.0f;
        camera.pitch = clamp_float(camera.pitch, -1.25f, 1.25f);
        camera.distance = clamp_float(camera.distance, kCameraMinDistance, kCameraMaxDistance);

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
            if (recreate_simulation(&vao, &vbo, new_count, mode, clustered, radius_profile)) {
                object_count = new_count;
                memset(&metrics, 0, sizeof(metrics));
            }
        }

        Vector2 mouse = GetMousePosition();
        const int mouse_active =
            mouse.x >= 0.0f && mouse.x < (float)kWindowWidth
            && mouse.y >= 0.0f && mouse.y < (float)kWindowHeight;

        if (!paused) {
            cuda_visualizer_step(frame_dt, mouse.x, mouse.y, mouse_active, &camera, &metrics);
        }

        BeginDrawing();
        ClearBackground((Color){8, 10, 14, 255});
        draw_container_box(1, &camera);
        draw_particles(vao, particle_shader, object_count);
        draw_container_box(0, &camera);
        if (mouse_active) {
            DrawCircleLines((int)mouse.x, (int)mouse.y, kMouseColliderRadius, (Color){255, 255, 255, 95});
            DrawCircle((int)mouse.x, (int)mouse.y, 3.0f, (Color){255, 255, 255, 160});
        }

        const char* compute_label =
            (mode == VISUALIZER_MODE_CPU_BRUTE_FORCE) ? "CPU compute" : "CUDA compute";

        DrawRectangle(12, 12, 520, 268, (Color){18, 22, 30, 220});
        DrawText("Raylib + CUDA-OpenGL Interop 3D", 24, 24, 20, RAYWHITE);
        DrawText(TextFormat("Objects: %d  [+/-]", object_count), 24, 52, 18, LIGHTGRAY);
        DrawText(TextFormat("Distribution: %s  [C]", clustered ? "clustered" : "uniform"), 24, 76, 18, LIGHTGRAY);
        DrawText(TextFormat("Radius: %s  [V]", visualizer_radius_profile_name(radius_profile)), 24, 100, 18, LIGHTGRAY);
        DrawText(TextFormat("Method: %s  [G]", mode_name(mode)), 24, 124, 18, LIGHTGRAY);
        DrawText(TextFormat("Collisions: %llu", metrics.collision_count), 24, 148, 18, LIGHTGRAY);
        DrawText(TextFormat("Candidate pairs: %llu", metrics.candidate_pair_count), 24, 172, 18, LIGHTGRAY);
        DrawText(TextFormat("%s: %.3f ms", compute_label, (double)metrics.gpu_time_ms), 24, 196, 18, LIGHTGRAY);
        DrawText(TextFormat("FPS: %d", GetFPS()), 24, 220, 18, LIGHTGRAY);
        DrawText(TextFormat("Camera: yaw %.1f pitch %.1f dist %.0f  [RMB/WASD/QE/F]",
                            (double)(camera.yaw * 57.29578f),
                            (double)(camera.pitch * 57.29578f),
                            (double)camera.distance),
                 24, 244, 18, LIGHTGRAY);

        EndDrawing();
    }

    cuda_visualizer_destroy();
    rlUnloadVertexBuffer(vbo);
    rlUnloadVertexArray(vao);
    rlUnloadShaderProgram(particle_shader);
    CloseWindow();
    return 0;
}
