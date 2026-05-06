#include "CpuBruteForceDetector.h"
#include "CudaBruteForceDetector.h"
#include "CudaGridDetector.h"
#include "CudaLBVHDetector.h"
#include "DataGenerator.h"

#include <raylib.h>
#include <rlgl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

struct RenderVertex {
    float x, y, local_x, local_y, r, g, b;
};

// Application State
static const int kWindowWidth = 1280;
static const int kWindowHeight = 720;
static int g_object_count = 2500;
static CirclesSoA g_circles;
static std::vector<float> g_vx;
static std::vector<float> g_vy;

enum class Method { CPU_BRUTE, CUDA_BRUTE, CUDA_GRID, CUDA_LBVH };
static Method g_current_method = Method::CUDA_LBVH;
static std::unique_ptr<ICollisionDetector> g_detector;

static cudaGraphicsResource* g_vbo_resource = nullptr;
static unsigned int g_vao = 0;
static unsigned int g_vbo = 0;

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

static void reset_simulation() {
    BenchmarkConfig config = benchmark_default_config();
    config.scene_width = kWindowWidth;
    config.scene_height = kWindowHeight;
    g_circles = generate_uniform_circles(g_object_count, config);
    
    g_vx.resize(g_object_count);
    g_vy.resize(g_object_count);
    
    for (int i = 0; i < g_object_count; ++i) {
        g_vx[i] = (static_cast<float>(rand()) / RAND_MAX) * 180.0f - 90.0f;
        g_vy[i] = (static_cast<float>(rand()) / RAND_MAX) * 180.0f - 90.0f;
    }
}

static void switch_detector(Method m) {
    g_current_method = m;
    switch (m) {
        case Method::CPU_BRUTE: g_detector = std::make_unique<CpuBruteForceDetector>(); break;
        case Method::CUDA_BRUTE: g_detector = std::make_unique<CudaBruteForceDetector>(); break;
        case Method::CUDA_GRID: {
            auto grid = std::make_unique<CudaGridDetector>();
            grid->set_grid_params(kWindowWidth, kWindowHeight, 20.0f, 128);
            g_detector = std::move(grid);
            break;
        }
        case Method::CUDA_LBVH: {
            auto lbvh = std::make_unique<CudaLBVHDetector>();
            lbvh->set_scene_bounds(kWindowWidth, kWindowHeight);
            g_detector = std::move(lbvh);
            break;
        }
    }
}

static void integrate_cpu(float dt) {
    for (int i = 0; i < g_object_count; ++i) {
        float& x = g_circles.host_x[i];
        float& y = g_circles.host_y[i];
        float& vx = g_vx[i];
        float& vy = g_vy[i];
        float r = g_circles.host_radius[i];

        x += vx * dt;
        y += vy * dt;

        if (x < r) { x = r; vx = std::abs(vx); }
        else if (x > kWindowWidth - r) { x = kWindowWidth - r; vx = -std::abs(vx); }
        if (y < r) { y = r; vy = std::abs(vy); }
        else if (y > kWindowHeight - r) { y = kWindowHeight - r; vy = -std::abs(vy); }
    }
}

static void generate_vbo_cpu(RenderVertex* vertices) {
    for (int i = 0; i < g_object_count; ++i) {
        float cx = (g_circles.host_x[i] / kWindowWidth) * 2.0f - 1.0f;
        float cy = 1.0f - (g_circles.host_y[i] / kWindowHeight) * 2.0f;
        float rx = std::max(9.0f, g_circles.host_radius[i] * 2.0f) / kWindowWidth * 2.0f;
        float ry = std::max(9.0f, g_circles.host_radius[i] * 2.0f) / kWindowHeight * 2.0f;

        float r = 0.18f, g = 0.78f, b = 1.0f;
        // Without collision highlights for CPU fallback rendering
        
        const float corners[6][2] = {{-1,-1},{1,-1},{1,1},{-1,-1},{1,1},{-1,1}};
        int base = i * 6;
        for (int v = 0; v < 6; ++v) {
            vertices[base+v] = { cx + corners[v][0]*rx, cy + corners[v][1]*ry, corners[v][0], corners[v][1], r, g, b };
        }
    }
}

__global__ void write_vbo_kernel_soa(
    const float* x, const float* y, const float* r,
    RenderVertex* vertices, int count, int width, int height) 
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float cx = (x[i] / width) * 2.0f - 1.0f;
    float cy = 1.0f - (y[i] / height) * 2.0f;
    float rx = fmaxf(9.0f, r[i] * 2.0f) / width * 2.0f;
    float ry = fmaxf(9.0f, r[i] * 2.0f) / height * 2.0f;

    // Greenish default
    float cr = 0.18f, cg = 0.78f, cb = 1.0f;

    const float corners[6][2] = {{-1,-1},{1,-1},{1,1},{-1,-1},{1,1},{-1,1}};
    int base = i * 6;
    for (int v = 0; v < 6; ++v) {
        vertices[base+v] = { cx + corners[v][0]*rx, cy + corners[v][1]*ry, corners[v][0], corners[v][1], cr, cg, cb };
    }
}

int run_application(int argc, char** argv) {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(kWindowWidth, kWindowHeight, "Broad-Phase Collision Detection - CUDA Refactor Demo");
    SetTargetFPS(60);

    unsigned int particle_shader = rlLoadShaderProgram(kVertexShader, kFragmentShader);
    
    // Create VBO
    g_vao = rlLoadVertexArray();
    rlEnableVertexArray(g_vao);
    g_vbo = rlLoadVertexBuffer(NULL, g_object_count * 6 * sizeof(RenderVertex), true);
    rlEnableVertexBuffer(g_vbo);
    rlSetVertexAttribute(0, 2, RL_FLOAT, false, sizeof(RenderVertex), 0);
    rlEnableVertexAttribute(0);
    rlSetVertexAttribute(1, 2, RL_FLOAT, false, sizeof(RenderVertex), 2 * sizeof(float));
    rlEnableVertexAttribute(1);
    rlSetVertexAttribute(2, 3, RL_FLOAT, false, sizeof(RenderVertex), 4 * sizeof(float));
    rlEnableVertexAttribute(2);
    rlDisableVertexBuffer();
    rlDisableVertexArray();

    cudaGraphicsGLRegisterBuffer(&g_vbo_resource, g_vbo, cudaGraphicsMapFlagsWriteDiscard);

    reset_simulation();
    switch_detector(Method::CUDA_LBVH);

    std::vector<RenderVertex> host_vertices(g_object_count * 6);
    CollisionResult last_result;
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) reset_simulation();
        if (IsKeyPressed(KEY_ONE)) switch_detector(Method::CPU_BRUTE);
        if (IsKeyPressed(KEY_TWO)) switch_detector(Method::CUDA_BRUTE);
        if (IsKeyPressed(KEY_THREE)) switch_detector(Method::CUDA_GRID);
        if (IsKeyPressed(KEY_FOUR)) switch_detector(Method::CUDA_LBVH);

        if (!paused) {
            integrate_cpu(GetFrameTime());
            
            g_detector->update_data(g_circles);
            last_result = g_detector->run_detection();

            // Zero-copy rendering for GPU, explicit CPU mapping for CPU
            if (g_current_method == Method::CPU_BRUTE) {
                generate_vbo_cpu(host_vertices.data());
                rlUpdateVertexBuffer(g_vbo, host_vertices.data(), host_vertices.size() * sizeof(RenderVertex), 0);
            } else {
                RenderVertex* d_vertices = nullptr;
                size_t mapped_size = 0;
                cudaGraphicsMapResources(1, &g_vbo_resource, 0);
                cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &mapped_size, g_vbo_resource);
                
                // Hack: We need device pointers from the detector, but we can just use the fact that 
                // g_circles has device pointers if we mapped them, or just upload to a quick local buffer
                // For a perfect SoA zero-copy we write VBO kernel here.
                float *d_x, *d_y, *d_r;
                cudaMalloc(&d_x, g_object_count * sizeof(float));
                cudaMalloc(&d_y, g_object_count * sizeof(float));
                cudaMalloc(&d_r, g_object_count * sizeof(float));
                cudaMemcpy(d_x, g_circles.host_x.data(), g_object_count * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_y, g_circles.host_y.data(), g_object_count * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_r, g_circles.host_radius.data(), g_object_count * sizeof(float), cudaMemcpyHostToDevice);

                int blocks = (g_object_count + 255) / 256;
                write_vbo_kernel_soa<<<blocks, 256>>>(d_x, d_y, d_r, d_vertices, g_object_count, kWindowWidth, kWindowHeight);

                cudaGraphicsUnmapResources(1, &g_vbo_resource, 0);
                
                cudaFree(d_x); cudaFree(d_y); cudaFree(d_r);
            }
        }

        BeginDrawing();
        ClearBackground(Color{8, 10, 14, 255});
        
        rlDrawRenderBatchActive();
        rlEnableShader(particle_shader);
        if (rlEnableVertexArray(g_vao)) {
            rlDrawVertexArray(0, g_object_count * 6);
            rlDisableVertexArray();
        }
        rlDisableShader();

        DrawRectangle(12, 12, 400, 240, Color{18, 22, 30, 220});
        DrawText("Collision Visualizer (1-4 to change)", 24, 24, 18, RAYWHITE);
        DrawText(TextFormat("Method: %s", g_detector->get_name().c_str()), 24, 52, 16, GREEN);
        DrawText(TextFormat("Objects: %d", g_object_count), 24, 76, 16, LIGHTGRAY);
        DrawText(TextFormat("Collisions: %llu", last_result.collision_count), 24, 100, 16, LIGHTGRAY);
        DrawText(TextFormat("Total Time: %.3f ms", last_result.total_time_ms), 24, 124, 16, RED);
        DrawText(TextFormat("Transfer Time: %.3f ms", last_result.memory_transfer_time_ms), 24, 148, 16, ORANGE);
        DrawText(TextFormat("Kernel Time: %.3f ms", last_result.kernel_execution_time_ms), 24, 172, 16, ORANGE);
        DrawText("SPACE: Pause | R: Reset", 24, 210, 16, DARKGRAY);

        EndDrawing();
    }

    cudaGraphicsUnregisterResource(g_vbo_resource);
    rlUnloadVertexBuffer(g_vbo);
    rlUnloadVertexArray(g_vao);
    rlUnloadShaderProgram(particle_shader);
    CloseWindow();
    return 0;
}
