#include <jni.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <android/log.h>

#define LOG_TAG "BrushBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#pragma pack(push, 1)
struct SplatEntry {
    float x, y, z;
    float s0, s1, s2;
    uint8_t r, g, b, a;
    uint8_t q0, q1, q2, q3;
};
#pragma pack(pop)

enum TrainExitCode {
    TrainSuccess = 0,
    TrainError = 1
};

struct ProgressMessage {
    uint32_t tag; // 0 = NewProcess, 1 = Training { iter }, 2 = DoneTraining
    uint32_t iter;
};

struct TrainOptions {
    uint32_t total_train_steps;
    uint32_t refine_every;
    uint32_t max_resolution;
    uint32_t export_every;
    const char* output_path;
};

typedef void (*ProgressCallback)(ProgressMessage msg, void* user_data);

typedef TrainExitCode (*TrainAndSaveFn)(
    const char* dataset_path,
    const TrainOptions* options,
    ProgressCallback progress_callback,
    void* user_data
);

// Global context for progress callback to Java/Kotlin
static JavaVM* g_jvm = nullptr;
static jobject g_listener_global = nullptr;
static jmethodID g_listener_mid = nullptr;
static uint32_t g_total_steps = 7000;

extern "C" void native_progress_callback(ProgressMessage msg, void* user_data) {
    if (msg.tag == 1) {
        if (msg.iter % 20 == 0 || msg.iter == g_total_steps) {
            float pct = (float)msg.iter / (float)(g_total_steps > 0 ? g_total_steps : 1);
            if (pct > 1.0f) pct = 1.0f;
            LOGI("On-Device 3DGS Progress: Step %u / %u (%.1f%%)", msg.iter, g_total_steps, pct * 100.0f);

            if (g_jvm && g_listener_global && g_listener_mid) {
                JNIEnv* env = nullptr;
                jint attach_res = g_jvm->AttachCurrentThread(&env, nullptr);
                if (attach_res == JNI_OK && env) {
                    env->CallVoidMethod(g_listener_global, g_listener_mid, (jint)msg.iter, (jfloat)pct);
                }
            }
        }
    } else if (msg.tag == 2) {
        LOGI("On-Device 3DGS Training finished successfully.");
    }
}

namespace {

TrainAndSaveFn resolve_train_and_save() {
    // 1. Try RTLD_DEFAULT (libbrush_c.so is preloaded via System.loadLibrary)
    TrainAndSaveFn fn = (TrainAndSaveFn)dlsym(RTLD_DEFAULT, "train_and_save");
    if (fn) {
        LOGI("Successfully resolved train_and_save via RTLD_DEFAULT");
        return fn;
    }

    // 2. Explicit dlopen of libbrush_c.so
    void* handle = dlopen("libbrush_c.so", RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
        fn = (TrainAndSaveFn)dlsym(handle, "train_and_save");
        if (fn) {
            LOGI("Successfully resolved train_and_save via dlopen(libbrush_c.so)");
            return fn;
        }
    }

    LOGE("Could not locate train_and_save: %s", dlerror());
    return nullptr;
}

// Converts a binary PLY from brush into a 32-byte GPU .splat buffer
bool convert_ply_to_splat(const std::string& ply_path, const std::string& splat_path) {
    LOGI("Converting PLY to 32-byte .splat: %s -> %s", ply_path.c_str(), splat_path.c_str());
    std::ifstream file(ply_path, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Failed to open PLY file: %s", ply_path.c_str());
        return false;
    }

    std::string line;
    int num_vertices = 0;
    std::vector<std::string> prop_names;
    std::vector<int> prop_sizes;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element") {
            std::string elem_type;
            iss >> elem_type;
            if (elem_type == "vertex") {
                iss >> num_vertices;
            }
        } else if (token == "property") {
            std::string type, name;
            iss >> type >> name;
            prop_names.push_back(name);
            if (type == "float" || type == "float32" || type == "int" || type == "uint") {
                prop_sizes.push_back(4);
            } else if (type == "double" || type == "float64") {
                prop_sizes.push_back(8);
            } else if (type == "uchar" || type == "uint8" || type == "char" || type == "int8") {
                prop_sizes.push_back(1);
            } else if (type == "short" || type == "uint16" || type == "int16") {
                prop_sizes.push_back(2);
            } else {
                prop_sizes.push_back(4);
            }
        } else if (token == "end_header") {
            break;
        }
    }

    if (num_vertices <= 0) {
        LOGE("Invalid PLY vertex count: %d", num_vertices);
        return false;
    }

    int stride = 0;
    int offset_x = -1, offset_y = -1, offset_z = -1;
    int offset_s0 = -1, offset_s1 = -1, offset_s2 = -1;
    int offset_f0 = -1, offset_f1 = -1, offset_f2 = -1;
    int offset_op = -1;
    int offset_r0 = -1, offset_r1 = -1, offset_r2 = -1, offset_r3 = -1;

    for (size_t i = 0; i < prop_names.size(); ++i) {
        const std::string& name = prop_names[i];
        if (name == "x") offset_x = stride;
        else if (name == "y") offset_y = stride;
        else if (name == "z") offset_z = stride;
        else if (name == "scale_0") offset_s0 = stride;
        else if (name == "scale_1") offset_s1 = stride;
        else if (name == "scale_2") offset_s2 = stride;
        else if (name == "f_dc_0") offset_f0 = stride;
        else if (name == "f_dc_1") offset_f1 = stride;
        else if (name == "f_dc_2") offset_f2 = stride;
        else if (name == "opacity") offset_op = stride;
        else if (name == "rot_0") offset_r0 = stride;
        else if (name == "rot_1") offset_r1 = stride;
        else if (name == "rot_2") offset_r2 = stride;
        else if (name == "rot_3") offset_r3 = stride;
        stride += prop_sizes[i];
    }

    std::ofstream out(splat_path, std::ios::binary);
    if (!out.is_open()) {
        LOGE("Failed to open output splat file: %s", splat_path.c_str());
        return false;
    }

    std::vector<char> v_buf(stride);
    const float C0 = 0.28209479177387814f;

    for (int i = 0; i < num_vertices; ++i) {
        file.read(v_buf.data(), stride);
        if (!file) break;

        float x = (offset_x >= 0) ? *(float*)(v_buf.data() + offset_x) : 0.0f;
        float y = (offset_y >= 0) ? *(float*)(v_buf.data() + offset_y) : 0.0f;
        float z = (offset_z >= 0) ? *(float*)(v_buf.data() + offset_z) : 0.0f;

        float s0 = (offset_s0 >= 0) ? std::exp(*(float*)(v_buf.data() + offset_s0)) : 0.01f;
        float s1 = (offset_s1 >= 0) ? std::exp(*(float*)(v_buf.data() + offset_s1)) : 0.01f;
        float s2 = (offset_s2 >= 0) ? std::exp(*(float*)(v_buf.data() + offset_s2)) : 0.01f;

        float f_dc_0 = (offset_f0 >= 0) ? *(float*)(v_buf.data() + offset_f0) : 0.0f;
        float f_dc_1 = (offset_f1 >= 0) ? *(float*)(v_buf.data() + offset_f1) : 0.0f;
        float f_dc_2 = (offset_f2 >= 0) ? *(float*)(v_buf.data() + offset_f2) : 0.0f;

        float r_f = (0.5f + C0 * f_dc_0) * 255.0f;
        float g_f = (0.5f + C0 * f_dc_1) * 255.0f;
        float b_f = (0.5f + C0 * f_dc_2) * 255.0f;
        uint8_t r = (uint8_t)std::max(0.0f, std::min(255.0f, r_f));
        uint8_t g = (uint8_t)std::max(0.0f, std::min(255.0f, g_f));
        uint8_t b = (uint8_t)std::max(0.0f, std::min(255.0f, b_f));

        float raw_op = (offset_op >= 0) ? *(float*)(v_buf.data() + offset_op) : 0.0f;
        float op = (1.0f / (1.0f + std::exp(-raw_op))) * 255.0f;
        uint8_t a = (uint8_t)std::max(0.0f, std::min(255.0f, op));

        float r0 = (offset_r0 >= 0) ? *(float*)(v_buf.data() + offset_r0) : 1.0f;
        float r1 = (offset_r1 >= 0) ? *(float*)(v_buf.data() + offset_r1) : 0.0f;
        float r2 = (offset_r2 >= 0) ? *(float*)(v_buf.data() + offset_r2) : 0.0f;
        float r3 = (offset_r3 >= 0) ? *(float*)(v_buf.data() + offset_r3) : 0.0f;
        float norm = std::sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3) + 1e-8f;
        r0 /= norm; r1 /= norm; r2 /= norm; r3 /= norm;

        uint8_t q0 = (uint8_t)std::max(0.0f, std::min(255.0f, r0 * 128.0f + 128.0f));
        uint8_t q1 = (uint8_t)std::max(0.0f, std::min(255.0f, r1 * 128.0f + 128.0f));
        uint8_t q2 = (uint8_t)std::max(0.0f, std::min(255.0f, r2 * 128.0f + 128.0f));
        uint8_t q3 = (uint8_t)std::max(0.0f, std::min(255.0f, r3 * 128.0f + 128.0f));

        SplatEntry entry = { x, y, z, s0, s1, s2, r, g, b, a, q0, q1, q2, q3 };
        out.write((const char*)&entry, sizeof(entry));
    }

    LOGI("Successfully converted %d Gaussians into %s (%d bytes)", num_vertices, splat_path.c_str(), num_vertices * 32);
    return true;
}

// Find newest .ply file in export directory
std::string find_latest_ply(const std::string& dir_path) {
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return "";
    struct dirent* entry;
    std::string latest_file = "";
    time_t latest_mtime = 0;

    while ((entry = readdir(dir)) != nullptr) {
        std::string fname = entry->d_name;
        if (fname.size() > 4 && fname.substr(fname.size() - 4) == ".ply") {
            std::string full_path = dir_path + "/" + fname;
            struct stat st;
            if (stat(full_path.c_str(), &st) == 0) {
                if (st.st_mtime > latest_mtime) {
                    latest_mtime = st.st_mtime;
                    latest_file = full_path;
                }
            }
        }
    }
    closedir(dir);
    return latest_file;
}

} // namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeTrainAndSave(
    JNIEnv *env, jobject thiz,
    jstring dataset_path, jstring output_path, jint iterations, jint max_resolution, jobject progress_listener
) {
    if (dataset_path == nullptr || output_path == nullptr) {
        LOGE("Invalid null paths provided to nativeTrainAndSave");
        return (jint)TrainError;
    }

    TrainAndSaveFn train_fn = resolve_train_and_save();
    if (!train_fn) {
        LOGE("Cannot proceed with on-device training: native symbol train_and_save is missing");
        return (jint)TrainError;
    }

    env->GetJavaVM(&g_jvm);

    if (progress_listener != nullptr) {
        g_listener_global = env->NewGlobalRef(progress_listener);
        jclass listener_class = env->GetObjectClass(progress_listener);
        g_listener_mid = env->GetMethodID(listener_class, "onProgress", "(IF)V");
    } else {
        g_listener_global = nullptr;
        g_listener_mid = nullptr;
    }

    const char* c_dataset = env->GetStringUTFChars(dataset_path, nullptr);
    const char* c_output = env->GetStringUTFChars(output_path, nullptr);

    g_total_steps = (uint32_t)iterations;
    std::string dataset_dir_str = c_dataset;
    std::string export_dir = dataset_dir_str + "/exports";

    // Ensure export directory exists
    mkdir(export_dir.c_str(), 0777);

    LOGI(">>> Starting On-Device 3DGS Optimization <<<");
    LOGI("Dataset Path: %s", c_dataset);
    LOGI("Target Output: %s", c_output);
    LOGI("Export Dir: %s", export_dir.c_str());
    LOGI("Steps: %d, Max Resolution: %d", iterations, max_resolution);

    TrainOptions options;
    options.total_train_steps = (uint32_t)iterations;
    options.refine_every = 100;
    options.max_resolution = (uint32_t)(max_resolution > 0 ? max_resolution : 720);
    options.export_every = (uint32_t)iterations;
    options.output_path = export_dir.c_str();

    TrainExitCode result = train_fn(
        c_dataset,
        &options,
        native_progress_callback,
        nullptr
    );

    LOGI("On-Device Training finished with code: %d", (int)result);

    // If training succeeded, locate the output PLY and convert to 32-byte .splat
    if (result == TrainSuccess) {
        std::string latest_ply = find_latest_ply(export_dir);
        if (latest_ply.empty()) {
            // Also search in dataset dir
            latest_ply = find_latest_ply(dataset_dir_str);
        }

        if (!latest_ply.empty()) {
            LOGI("Found trained checkpoint PLY: %s", latest_ply.c_str());
            std::string final_out = c_output;
            if (final_out.size() > 4 && final_out.substr(final_out.size() - 4) == ".ply") {
                // If output path is requested as PLY, copy directly
                std::ifstream src(latest_ply, std::ios::binary);
                std::ofstream dst(final_out, std::ios::binary);
                dst << src.rdbuf();
            } else {
                // Default: convert to 32-byte binary .splat for 60 FPS mobile rendering
                convert_ply_to_splat(latest_ply, final_out);
            }
        } else {
            LOGE("Warning: No .ply checkpoint found in %s after training.", export_dir.c_str());
        }
    }

    if (g_listener_global && env) {
        env->DeleteGlobalRef(g_listener_global);
        g_listener_global = nullptr;
        g_listener_mid = nullptr;
    }

    env->ReleaseStringUTFChars(dataset_path, c_dataset);
    env->ReleaseStringUTFChars(output_path, c_output);

    return (jint)result;
}

} // extern "C"
