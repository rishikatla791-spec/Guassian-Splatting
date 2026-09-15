// JNI bridge to the prebuilt Rust/wgpu training engine (libbrush_c.so).
//
// What the engine's C ABI actually gives us (brush/apps/brush-c/src/lib.rs):
//
//   TrainExitCode train_and_save(const char* dataset_path,
//                                const TrainOptions*,
//                                ProgressCallback, void* user_data);
//
//   struct TrainOptions { u32 total_train_steps; u32 refine_every;
//                         u32 max_resolution; u32 export_every;
//                         const char* output_path; };
//
//   enum TrainExitCode { Success = 0, Error = 1 };
//
// That is the *entire* surface: `llvm-readelf --dyn-syms libbrush_c.so` shows
// exactly one usable export, `train_and_save`. Consequences we have to live
// with, and which this file works around:
//
//  * There is no error string. Internal `anyhow` errors are matched as
//    `Err(_) => TrainExitCode::Error` and dropped on the floor.
//  * brush logs through the Rust `log` facade but `brush-c` never installs a
//    logger, and the .so does not even import `__android_log_write`, so none of
//    its logging can reach logcat. Panics, however, are printed to stderr by
//    the default panic hook before `catch_unwind` swallows them -- so we
//    redirect stderr/stdout into logcat and keep a tail of it (see
//    start_log_pump) which is the only channel that carries a real reason.
//  * There is no cancellation entry point (see request_cancel).
//  * There is no max-splat / Gaussian-cap field (see convert_ply_to_splat).
//
// Everything else here exists to make a failure *diagnosable* rather than a
// bare "exit code 1".

#include <jni.h>
#include <string>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <android/log.h>
#include <vulkan/vulkan.h>

#define LOG_TAG "BrushBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#pragma pack(push, 1)
struct SplatEntry {
    float x, y, z;
    float s0, s1, s2;
    uint8_t r, g, b, a;
    uint8_t q0, q1, q2, q3;
};
#pragma pack(pop)
static_assert(sizeof(SplatEntry) == 32, ".splat record must be exactly 32 bytes");

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

// ---------------------------------------------------------------------------
// Global state & synchronization
// ---------------------------------------------------------------------------

static JavaVM* g_jvm = nullptr;
static jobject g_listener_global = nullptr;
static jmethodID g_listener_mid = nullptr;
static jmethodID g_listener_state_mid = nullptr;
static std::mutex g_callback_mutex;

static std::atomic<bool> g_cancel_requested{false};
static std::atomic<bool> g_engine_poisoned{false};   // a cancelled run parked the engine thread
static std::atomic<bool> g_run_active{false};
static std::atomic<uint32_t> g_total_steps{7000};

// Thermal / battery duty cycling. Driven from Kotlin (PowerManager lives there);
// the actual sleeping has to happen here because the progress callback runs on
// the very thread that drives training.
static std::atomic<int>  g_throttle_ms{0};
static std::atomic<bool> g_paused{false};

// Where the run got to, so a bare exit code 1 can be attributed to a phase.
enum TrainPhase {
    PHASE_NOT_STARTED = 0,   // train_and_save returned before emitting anything
    PHASE_PROCESS_STARTED = 1,
    PHASE_TRAINING = 2,
    PHASE_DONE_TRAINING = 3
};
static std::atomic<int> g_phase{PHASE_NOT_STARTED};
static std::atomic<uint32_t> g_last_iter{0};

// ---------------------------------------------------------------------------
// stdout/stderr -> logcat pump
// ---------------------------------------------------------------------------
// libbrush_c.so has no logger installed and does not link liblog, so the ONLY
// thing that ever escapes it is what Rust's default panic hook writes to
// stderr. On Android stderr goes to /dev/null, which is why "exit code 1" has
// been arriving with no reason attached. Redirect both fds into a reader
// thread, mirror every line into logcat, and keep a tail to attach to the
// error we hand back to Kotlin.

static std::mutex g_tail_mutex;
static std::deque<std::string> g_log_tail;
static const size_t kLogTailLines = 120;

static void push_log_tail(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_tail_mutex);
    g_log_tail.push_back(line);
    while (g_log_tail.size() > kLogTailLines) g_log_tail.pop_front();
}

static void clear_log_tail() {
    std::lock_guard<std::mutex> lock(g_tail_mutex);
    g_log_tail.clear();
}

static std::string log_tail_text(size_t max_lines, size_t max_chars) {
    std::lock_guard<std::mutex> lock(g_tail_mutex);
    std::vector<std::string> picked;
    size_t start = g_log_tail.size() > max_lines ? g_log_tail.size() - max_lines : 0;
    for (size_t i = start; i < g_log_tail.size(); ++i) picked.push_back(g_log_tail[i]);
    std::string out;
    for (size_t i = 0; i < picked.size(); ++i) {
        if (!out.empty()) out += "\n";
        out += picked[i];
    }
    if (out.size() > max_chars) out = out.substr(out.size() - max_chars);
    return out;
}

static int g_log_pipe[2] = {-1, -1};

static void start_log_pump() {
    static std::once_flag once;
    std::call_once(once, []() {
        // Rust reads these at panic time; brush-c never installs a `log` sink so
        // RUST_LOG alone does nothing, but the backtrace is pure profit.
        setenv("RUST_BACKTRACE", "full", 1);
        setenv("RUST_LOG", "info", 1);

        setvbuf(stdout, nullptr, _IOLBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);

        if (pipe(g_log_pipe) != 0) {
            LOGW("Could not create log pipe (%s); native engine output stays invisible", strerror(errno));
            return;
        }
        dup2(g_log_pipe[1], STDOUT_FILENO);
        dup2(g_log_pipe[1], STDERR_FILENO);

        std::thread([]() {
            char buf[1024];
            std::string pending;
            ssize_t n;
            while ((n = read(g_log_pipe[0], buf, sizeof(buf) - 1)) > 0) {
                pending.append(buf, static_cast<size_t>(n));
                size_t pos;
                while ((pos = pending.find('\n')) != std::string::npos) {
                    std::string line = pending.substr(0, pos);
                    pending.erase(0, pos + 1);
                    if (line.empty()) continue;
                    __android_log_write(ANDROID_LOG_INFO, "BrushEngine", line.c_str());
                    push_log_tail(line);
                }
                if (pending.size() > 4096) {   // a line with no newline in sight
                    __android_log_write(ANDROID_LOG_INFO, "BrushEngine", pending.c_str());
                    push_log_tail(pending);
                    pending.clear();
                }
            }
        }).detach();

        LOGI("Native stdout/stderr are now mirrored to logcat tag 'BrushEngine'");
    });
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

namespace {

std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    out += b;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

long long file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return -1;
    return static_cast<long long>(st.st_size);
}

bool copy_file(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) return false;
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out << in.rdbuf();
    out.flush();
    return out.good();
}

struct PlyCandidate {
    std::string path;
    time_t mtime = 0;
    uint32_t iter = 0;
    long long bytes = 0;
    bool valid() const { return !path.empty(); }
};

uint32_t iter_from_export_name(const std::string& fname) {
    // "export_01500.ply" -> 1500
    size_t us = fname.rfind('_');
    if (us == std::string::npos) return 0;
    size_t dot = fname.rfind('.');
    if (dot == std::string::npos || dot <= us + 1) return 0;
    std::string digits = fname.substr(us + 1, dot - us - 1);
    for (char c : digits) if (!isdigit(static_cast<unsigned char>(c))) return 0;
    return static_cast<uint32_t>(strtoul(digits.c_str(), nullptr, 10));
}

/**
 * Newest .ply in `dir_path`. `min_mtime` lets the caller demand a file this run
 * actually produced -- without it, a stale checkpoint from an earlier run gets
 * picked up and handed to the user as a fresh result.
 */
PlyCandidate find_latest_ply(const std::string& dir_path, time_t min_mtime) {
    PlyCandidate best;
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return best;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fname = entry->d_name;
        if (fname.size() <= 4 || fname.substr(fname.size() - 4) != ".ply") continue;
        std::string full_path = dir_path + "/" + fname;
        struct stat st;
        if (stat(full_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (st.st_size <= 0) continue;
        if (min_mtime > 0 && st.st_mtime < min_mtime) continue;
        if (best.path.empty() || st.st_mtime > best.mtime ||
            (st.st_mtime == best.mtime && iter_from_export_name(fname) > best.iter)) {
            best.path = full_path;
            best.mtime = st.st_mtime;
            best.iter = iter_from_export_name(fname);
            best.bytes = static_cast<long long>(st.st_size);
        }
    }
    closedir(dir);
    return best;
}

void remove_plys_in(const std::string& dir_path) {
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return;
    struct dirent* entry;
    int removed = 0;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fname = entry->d_name;
        if (fname.size() > 4 && fname.substr(fname.size() - 4) == ".ply") {
            if (unlink((dir_path + "/" + fname).c_str()) == 0) removed++;
        }
    }
    closedir(dir);
    if (removed) LOGI("Cleared %d stale checkpoint(s) from %s", removed, dir_path.c_str());
}

/**
 * Copy a binary PLY, keeping at most `max_vertices` of them (strided).
 *
 * Used when staging a checkpoint as the warm-restart point cloud: a run that
 * over-densified can leave a checkpoint far larger than the device's budget,
 * and feeding that straight back in as the initial cloud would reproduce the
 * out-of-memory that killed the run in the first place. The engine subsamples
 * the initial cloud to `--max-splats` itself, but that knob defaults to
 * 10,000,000 and is not reachable through the C ABI.
 *
 * Property-agnostic: the header is rewritten with the new count and whole
 * fixed-size vertex records are copied, so every attribute survives.
 */
bool decimate_ply(const std::string& src, const std::string& dst, long max_vertices, long* out_kept) {
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) return false;

    std::string header_text;
    std::string line;
    long declared = 0;
    int stride = 0;
    bool binary_le = false;
    bool saw_format = false;
    size_t vertex_count_line = std::string::npos;

    while (std::getline(in, line)) {
        std::string raw = line;
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        std::istringstream iss(raw);
        std::string token;
        iss >> token;
        if (token == "format") {
            std::string fmt; iss >> fmt;
            saw_format = true;
            binary_le = (fmt == "binary_little_endian");
        } else if (token == "element") {
            std::string kind; iss >> kind;
            if (kind == "vertex") {
                iss >> declared;
                vertex_count_line = header_text.size();   // patch point
            }
        } else if (token == "property") {
            std::string type, name; iss >> type >> name;
            if (type == "double" || type == "float64") stride += 8;
            else if (type == "uchar" || type == "uint8" || type == "char" || type == "int8") stride += 1;
            else if (type == "short" || type == "uint16" || type == "int16") stride += 2;
            else stride += 4;
        }
        header_text += raw;
        header_text += "\n";
        if (token == "end_header") break;
    }

    if (!saw_format || !binary_le || declared <= 0 || stride <= 0 ||
        vertex_count_line == std::string::npos) {
        return false;
    }

    long step = 1;
    if (max_vertices > 0 && declared > max_vertices) {
        step = (declared + max_vertices - 1) / max_vertices;
    }
    long kept = (declared + step - 1) / step;

    // Patch "element vertex N" in place.
    size_t eol = header_text.find('\n', vertex_count_line);
    if (eol == std::string::npos) return false;
    header_text.replace(vertex_count_line, eol - vertex_count_line,
                        "element vertex " + std::to_string(kept));

    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out.write(header_text.data(), (std::streamsize)header_text.size());

    std::vector<char> rec((size_t)stride);
    long written = 0;
    for (long i = 0; i < declared; ++i) {
        in.read(rec.data(), stride);
        if (in.gcount() != stride) break;
        if (i % step != 0) continue;
        out.write(rec.data(), stride);
        written++;
    }
    out.flush();
    if (!out.good() || written == 0) return false;
    if (written != kept) {
        // The source was truncated, so the header we just wrote would lie about
        // the record count. Refuse rather than hand the engine a broken cloud.
        LOGW("Checkpoint %s is truncated (%ld of %ld records); not using it as a restart point",
             src.c_str(), written, kept);
        return false;
    }
    if (out_kept) *out_kept = written;
    if (step > 1) {
        LOGW("Decimated checkpoint %ld -> %ld vertices (1-in-%ld) to fit the device budget",
             declared, written, step);
    }
    return true;
}

TrainAndSaveFn resolve_train_and_save() {
    // 1. RTLD_DEFAULT (libbrush_c.so is normally preloaded via System.loadLibrary)
    TrainAndSaveFn fn = (TrainAndSaveFn)dlsym(RTLD_DEFAULT, "train_and_save");
    if (fn) return fn;

    // 2. Explicit dlopen
    void* handle = dlopen("libbrush_c.so", RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
        fn = (TrainAndSaveFn)dlsym(handle, "train_and_save");
        if (fn) return fn;
    }

    const char* err = dlerror();
    LOGE("Could not locate train_and_save: %s", err ? err : "(no dlerror)");
    return nullptr;
}

// ---------------------------------------------------------------------------
// Vulkan probe
// ---------------------------------------------------------------------------
// A Snapdragon 695 / Adreno 619 failing at ~15s is either a compute feature gap
// or memory, and until now we had no way to tell the two apart. Probing through
// dlopen (never linking libvulkan) keeps this harmless on devices/emulators
// that have no Vulkan at all.

std::string vulkan_report() {
    std::string out;
    void* lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return "vulkan: libvulkan.so not present";

    auto getProc = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    if (!getProc) { dlclose(lib); return "vulkan: vkGetInstanceProcAddr missing"; }

    auto pCreateInstance = (PFN_vkCreateInstance)getProc(nullptr, "vkCreateInstance");
    auto pEnumInstanceVersion =
        (PFN_vkEnumerateInstanceVersion)getProc(nullptr, "vkEnumerateInstanceVersion");
    if (!pCreateInstance) { dlclose(lib); return "vulkan: vkCreateInstance missing"; }

    uint32_t instance_version = VK_API_VERSION_1_0;
    if (pEnumInstanceVersion) pEnumInstanceVersion(&instance_version);

    char buf[512];
    snprintf(buf, sizeof(buf), "vulkan instance %u.%u.%u",
             VK_VERSION_MAJOR(instance_version),
             VK_VERSION_MINOR(instance_version),
             VK_VERSION_PATCH(instance_version));
    out += buf;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "mobile3dgs-probe";
    app.apiVersion = instance_version;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = pCreateInstance(&ici, nullptr, &instance);
    if (res != VK_SUCCESS || instance == VK_NULL_HANDLE) {
        snprintf(buf, sizeof(buf), "; vkCreateInstance failed (VkResult %d)", (int)res);
        out += buf;
        dlclose(lib);
        return out;
    }

    auto pEnumDevices    = (PFN_vkEnumeratePhysicalDevices)getProc(instance, "vkEnumeratePhysicalDevices");
    auto pGetProps       = (PFN_vkGetPhysicalDeviceProperties)getProc(instance, "vkGetPhysicalDeviceProperties");
    auto pGetMemProps    = (PFN_vkGetPhysicalDeviceMemoryProperties)getProc(instance, "vkGetPhysicalDeviceMemoryProperties");
    auto pGetFeatures    = (PFN_vkGetPhysicalDeviceFeatures)getProc(instance, "vkGetPhysicalDeviceFeatures");
    auto pGetProps2      = (PFN_vkGetPhysicalDeviceProperties2)getProc(instance, "vkGetPhysicalDeviceProperties2");
    auto pDestroy        = (PFN_vkDestroyInstance)getProc(instance, "vkDestroyInstance");

    if (pEnumDevices && pGetProps) {
        uint32_t count = 0;
        pEnumDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        if (count) pEnumDevices(instance, &count, devices.data());
        snprintf(buf, sizeof(buf), "; devices=%u", count);
        out += buf;

        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties props{};
            pGetProps(devices[i], &props);
            snprintf(buf, sizeof(buf),
                     "\n  [%u] %s (type %d) api %u.%u.%u driver 0x%x",
                     i, props.deviceName, (int)props.deviceType,
                     VK_VERSION_MAJOR(props.apiVersion),
                     VK_VERSION_MINOR(props.apiVersion),
                     VK_VERSION_PATCH(props.apiVersion),
                     props.driverVersion);
            out += buf;

            snprintf(buf, sizeof(buf),
                     "\n      limits: maxComputeWorkGroupInvocations=%u sharedMem=%u "
                     "maxStorageBufferRange=%uMB maxPerStageStorageBuffers=%u maxBoundSets=%u "
                     "maxComputeWorkGroupCount=[%u,%u,%u]",
                     props.limits.maxComputeWorkGroupInvocations,
                     props.limits.maxComputeSharedMemorySize,
                     (unsigned)(props.limits.maxStorageBufferRange / (1024u * 1024u)),
                     props.limits.maxPerStageDescriptorStorageBuffers,
                     props.limits.maxBoundDescriptorSets,
                     props.limits.maxComputeWorkGroupCount[0],
                     props.limits.maxComputeWorkGroupCount[1],
                     props.limits.maxComputeWorkGroupCount[2]);
            out += buf;

            if (pGetFeatures) {
                VkPhysicalDeviceFeatures feats{};
                pGetFeatures(devices[i], &feats);
                snprintf(buf, sizeof(buf),
                         "\n      features: shaderInt64=%d shaderFloat64=%d shaderInt16=%d "
                         "fragmentStoresAndAtomics=%d vertexStoresAndAtomics=%d "
                         "shaderStorageBufferArrayDynamicIndexing=%d",
                         (int)feats.shaderInt64, (int)feats.shaderFloat64, (int)feats.shaderInt16,
                         (int)feats.fragmentStoresAndAtomics,
                         (int)feats.vertexPipelineStoresAndAtomics,
                         (int)feats.shaderStorageBufferArrayDynamicIndexing);
                out += buf;
            }

            if (pGetProps2 && VK_VERSION_MINOR(props.apiVersion) >= 1) {
                VkPhysicalDeviceSubgroupProperties sub{};
                sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
                VkPhysicalDeviceProperties2 p2{};
                p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                p2.pNext = &sub;
                pGetProps2(devices[i], &p2);
                snprintf(buf, sizeof(buf),
                         "\n      subgroup: size=%u supportedStages=0x%x supportedOps=0x%x",
                         sub.subgroupSize, sub.supportedStages, sub.supportedOperations);
                out += buf;
            }

            if (pGetMemProps) {
                VkPhysicalDeviceMemoryProperties mem{};
                pGetMemProps(devices[i], &mem);
                unsigned long long device_local_mb = 0, host_visible_mb = 0;
                for (uint32_t h = 0; h < mem.memoryHeapCount; ++h) {
                    unsigned long long mb = mem.memoryHeaps[h].size / (1024ull * 1024ull);
                    if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) device_local_mb += mb;
                    else host_visible_mb += mb;
                }
                snprintf(buf, sizeof(buf), "\n      heaps: deviceLocal=%lluMB other=%lluMB",
                         device_local_mb, host_visible_mb);
                out += buf;
            }
        }
    }

    if (pDestroy) pDestroy(instance, nullptr);
    dlclose(lib);
    return out;
}

// ---------------------------------------------------------------------------
// PLY -> 32-byte .splat
// ---------------------------------------------------------------------------

struct ConvertResult {
    bool ok = false;
    long written = 0;
    long source_vertices = 0;
    bool capped = false;
    std::string error;
};

/**
 * Streaming binary-PLY -> 32-byte .splat converter.
 *
 * `max_gaussians` is the only place a Gaussian budget can be enforced: the C
 * ABI's TrainOptions has no max-splat field, so densification inside the engine
 * is genuinely unbounded (see the note in NativeBrushEngine.kt). Capping here
 * at least bounds what we ship and what the viewer has to load.
 */
ConvertResult convert_ply_to_splat(const std::string& ply_path,
                                   const std::string& splat_path,
                                   long max_gaussians) {
    ConvertResult r;
    LOGI("Streaming PLY to 32-byte .splat: %s -> %s", ply_path.c_str(), splat_path.c_str());
    std::ifstream file(ply_path, std::ios::binary);
    if (!file.is_open()) {
        r.error = "Could not open trained PLY: " + ply_path;
        return r;
    }

    std::string line;
    int num_vertices = 0;
    bool binary_le = false;
    bool saw_format = false;
    std::vector<std::string> prop_names;
    std::vector<int> prop_sizes;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "format") {
            std::string fmt;
            iss >> fmt;
            saw_format = true;
            binary_le = (fmt == "binary_little_endian");
        } else if (token == "element") {
            std::string elem_type;
            iss >> elem_type;
            if (elem_type == "vertex") iss >> num_vertices;
        } else if (token == "property") {
            std::string type, name;
            iss >> type >> name;
            prop_names.push_back(name);
            if (type == "float" || type == "float32" || type == "int" || type == "uint" ||
                type == "int32" || type == "uint32") {
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

    if (saw_format && !binary_le) {
        r.error = "Trained PLY is not binary_little_endian; refusing to guess its layout";
        return r;
    }
    if (num_vertices <= 0) {
        r.error = "Trained PLY declares " + std::to_string(num_vertices) + " vertices";
        return r;
    }
    r.source_vertices = num_vertices;

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

    if (stride <= 0 || offset_x < 0 || offset_y < 0 || offset_z < 0) {
        r.error = "Trained PLY has no x/y/z properties";
        return r;
    }

    // Deterministic strided decimation when the run blew past the device budget.
    long keep_step = 1;
    if (max_gaussians > 0 && static_cast<long>(num_vertices) > max_gaussians) {
        keep_step = (static_cast<long>(num_vertices) + max_gaussians - 1) / max_gaussians;
        r.capped = true;
        LOGW("Trained model has %d Gaussians but this device's budget is %ld; "
             "decimating 1-in-%ld. (The engine's C ABI has no max-splat option, "
             "so growth could not be bounded during training.)",
             num_vertices, max_gaussians, keep_step);
    }

    std::ofstream out(splat_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        r.error = "Could not open output for writing: " + splat_path;
        return r;
    }

    const size_t BATCH_SIZE = 4096;
    std::vector<char> raw_batch(static_cast<size_t>(stride) * BATCH_SIZE);
    std::vector<SplatEntry> splat_batch;
    splat_batch.reserve(BATCH_SIZE);

    const float C0 = 0.28209479177387814f;
    long total_written = 0;
    long vertices_remaining = num_vertices;
    long global_index = 0;

    while (vertices_remaining > 0) {
        size_t current_batch = std::min(static_cast<size_t>(vertices_remaining), BATCH_SIZE);
        file.read(raw_batch.data(), static_cast<std::streamsize>(stride * current_batch));
        size_t bytes_read = static_cast<size_t>(file.gcount());
        size_t actual_vertices = bytes_read / static_cast<size_t>(stride);
        if (actual_vertices == 0) break;

        splat_batch.clear();
        for (size_t i = 0; i < actual_vertices; ++i, ++global_index) {
            if (keep_step > 1 && (global_index % keep_step) != 0) continue;
            const char* v_data = raw_batch.data() + i * static_cast<size_t>(stride);

            float x, y, z;
            memcpy(&x, v_data + offset_x, 4);
            memcpy(&y, v_data + offset_y, 4);
            memcpy(&z, v_data + offset_z, 4);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

            auto readf = [&](int off, float fallback) -> float {
                if (off < 0) return fallback;
                float v;
                memcpy(&v, v_data + off, 4);
                return std::isfinite(v) ? v : fallback;
            };

            float s0 = std::exp(readf(offset_s0, -4.6f));
            float s1 = std::exp(readf(offset_s1, -4.6f));
            float s2 = std::exp(readf(offset_s2, -4.6f));
            if (!std::isfinite(s0) || !std::isfinite(s1) || !std::isfinite(s2)) continue;

            float f_dc_0 = readf(offset_f0, 0.0f);
            float f_dc_1 = readf(offset_f1, 0.0f);
            float f_dc_2 = readf(offset_f2, 0.0f);

            float r_f = (0.5f + C0 * f_dc_0) * 255.0f;
            float g_f = (0.5f + C0 * f_dc_1) * 255.0f;
            float b_f = (0.5f + C0 * f_dc_2) * 255.0f;
            uint8_t r8 = (uint8_t)std::max(0.0f, std::min(255.0f, r_f));
            uint8_t g8 = (uint8_t)std::max(0.0f, std::min(255.0f, g_f));
            uint8_t b8 = (uint8_t)std::max(0.0f, std::min(255.0f, b_f));

            float raw_op = readf(offset_op, 0.0f);
            float op = (1.0f / (1.0f + std::exp(-raw_op))) * 255.0f;
            uint8_t a8 = (uint8_t)std::max(0.0f, std::min(255.0f, op));

            float q_w = readf(offset_r0, 1.0f);
            float q_x = readf(offset_r1, 0.0f);
            float q_y = readf(offset_r2, 0.0f);
            float q_z = readf(offset_r3, 0.0f);
            float norm = std::sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z);
            if (!std::isfinite(norm) || norm < 1e-6f) { q_w = 1.0f; q_x = q_y = q_z = 0.0f; norm = 1.0f; }
            q_w /= norm; q_x /= norm; q_y /= norm; q_z /= norm;

            auto quant = [](float v) -> uint8_t {
                return (uint8_t)std::max(0.0f, std::min(255.0f, v * 128.0f + 128.0f));
            };

            splat_batch.push_back({ x, y, z, s0, s1, s2, r8, g8, b8, a8,
                                    quant(q_w), quant(q_x), quant(q_y), quant(q_z) });
        }

        if (!splat_batch.empty()) {
            out.write((const char*)splat_batch.data(),
                      (std::streamsize)(splat_batch.size() * sizeof(SplatEntry)));
            total_written += static_cast<long>(splat_batch.size());
        }

        vertices_remaining -= static_cast<long>(actual_vertices);
    }

    out.flush();
    if (!out.good()) {
        r.error = "Write failed while producing " + splat_path + " (out of storage?)";
        return r;
    }
    out.close();

    r.written = total_written;
    r.ok = total_written > 0;
    if (!r.ok) r.error = "Converted 0 Gaussians from a PLY declaring " + std::to_string(num_vertices);
    LOGI("Conversion complete: %ld Gaussians -> %s (%lld bytes)",
         total_written, splat_path.c_str(), (long long)total_written * 32);
    return r;
}

// ---------------------------------------------------------------------------
// .splat validation
// ---------------------------------------------------------------------------

struct SplatValidation {
    bool ok = false;
    long long bytes = 0;
    long count = 0;
    long non_finite = 0;
    long bad_quat = 0;
    long bad_scale = 0;
    long opaque = 0;      // alpha > 0
    std::string error;
};

/**
 * Structural + numeric check of a finished .splat. A corrupt or degenerate model
 * must fail loudly instead of being handed to the viewer as a result.
 */
SplatValidation validate_splat(const std::string& path, long max_gaussians) {
    SplatValidation v;
    v.bytes = file_size(path);
    if (v.bytes < 0) { v.error = "Output file does not exist: " + path; return v; }
    if (v.bytes == 0) { v.error = "Output file is empty"; return v; }
    if (v.bytes % 32 != 0) {
        v.error = "Output is " + std::to_string(v.bytes) + " bytes, not a multiple of the 32-byte record size (truncated write)";
        return v;
    }
    v.count = static_cast<long>(v.bytes / 32);
    if (v.count < 16) {
        v.error = "Only " + std::to_string(v.count) + " Gaussians in the output; nothing usable was produced";
        return v;
    }
    const long kAbsurd = 25000000L;
    if (v.count > kAbsurd) {
        v.error = "Implausible Gaussian count " + std::to_string(v.count);
        return v;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) { v.error = "Could not reopen output for validation"; return v; }

    const size_t kChunk = 4096;
    std::vector<SplatEntry> buf(kChunk);
    long remaining = v.count;
    while (remaining > 0) {
        size_t n = static_cast<size_t>(std::min<long>(remaining, static_cast<long>(kChunk)));
        in.read((char*)buf.data(), (std::streamsize)(n * sizeof(SplatEntry)));
        size_t got = static_cast<size_t>(in.gcount()) / sizeof(SplatEntry);
        if (got == 0) break;
        for (size_t i = 0; i < got; ++i) {
            const SplatEntry& e = buf[i];
            if (!std::isfinite(e.x) || !std::isfinite(e.y) || !std::isfinite(e.z) ||
                !std::isfinite(e.s0) || !std::isfinite(e.s1) || !std::isfinite(e.s2)) {
                v.non_finite++;
                continue;
            }
            if (e.s0 < 0.0f || e.s1 < 0.0f || e.s2 < 0.0f ||
                e.s0 > 1.0e4f || e.s1 > 1.0e4f || e.s2 > 1.0e4f) {
                v.bad_scale++;
            }
            float qw = (e.q0 - 128.0f) / 128.0f;
            float qx = (e.q1 - 128.0f) / 128.0f;
            float qy = (e.q2 - 128.0f) / 128.0f;
            float qz = (e.q3 - 128.0f) / 128.0f;
            float norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            // 8-bit quantisation costs ~0.004 of norm per component; 0.15 is slack.
            if (!(norm > 0.85f && norm < 1.15f)) v.bad_quat++;
            if (e.a > 0) v.opaque++;
        }
        remaining -= static_cast<long>(got);
    }

    if (remaining > 0) {
        v.error = "Output ended early during validation (" + std::to_string(remaining) + " records short)";
        return v;
    }
    if (v.non_finite > 0) {
        v.error = std::to_string(v.non_finite) + " Gaussians carry non-finite position/scale";
        return v;
    }
    if (v.bad_quat * 100 > v.count) {   // >1%
        v.error = std::to_string(v.bad_quat) + " of " + std::to_string(v.count) +
                  " rotations are not unit quaternions";
        return v;
    }
    if (v.bad_scale * 100 > v.count * 5) {   // >5%
        v.error = std::to_string(v.bad_scale) + " of " + std::to_string(v.count) +
                  " Gaussians have degenerate scales";
        return v;
    }
    if (v.opaque == 0) {
        v.error = "Every Gaussian is fully transparent; the model would render as nothing";
        return v;
    }
    if (max_gaussians > 0 && v.count > max_gaussians) {
        v.error = "Output holds " + std::to_string(v.count) +
                  " Gaussians, over this device's budget of " + std::to_string(max_gaussians);
        return v;
    }

    v.ok = true;
    return v;
}

std::string validation_json(const SplatValidation& v) {
    std::string s = "{";
    s += "\"ok\":" + std::string(v.ok ? "true" : "false");
    s += ",\"bytes\":" + std::to_string(v.bytes);
    s += ",\"count\":" + std::to_string(v.count);
    s += ",\"nonFinite\":" + std::to_string(v.non_finite);
    s += ",\"badQuat\":" + std::to_string(v.bad_quat);
    s += ",\"badScale\":" + std::to_string(v.bad_scale);
    s += ",\"opaque\":" + std::to_string(v.opaque);
    s += ",\"message\":\"" + json_escape(v.error) + "\"";
    s += "}";
    return s;
}

// ---------------------------------------------------------------------------
// Dataset preflight
// ---------------------------------------------------------------------------

std::string dataset_preflight(const std::string& dataset_dir) {
    struct stat st;
    if (stat(dataset_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
        return "Dataset directory does not exist: " + dataset_dir;

    std::string transforms = dataset_dir + "/transforms.json";
    bool has_transforms = file_exists(transforms);
    long long transforms_bytes = has_transforms ? file_size(transforms) : 0;

    std::string images_dir = dataset_dir + "/images";
    int image_count = 0;
    DIR* dir = opendir(images_dir.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string f = entry->d_name;
            if (f.size() > 4) {
                std::string ext = f.substr(f.size() - 4);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".png" || ext == "jpeg") image_count++;
            }
        }
        closedir(dir);
    }

    if (!has_transforms)
        return "Dataset has no transforms.json (found " + std::to_string(image_count) + " images)";
    if (transforms_bytes < 32)
        return "transforms.json is empty or truncated (" + std::to_string(transforms_bytes) + " bytes)";
    if (image_count < 4)
        return "Dataset has only " + std::to_string(image_count) +
               " images in " + images_dir + "; training needs several views";

    LOGI("Dataset preflight OK: %d images, transforms.json %lld bytes", image_count, transforms_bytes);
    return "";
}

// ---------------------------------------------------------------------------
// Kotlin notification helpers
// ---------------------------------------------------------------------------

struct JniScope {
    JNIEnv* env = nullptr;
    bool detach = false;
    explicit JniScope(JavaVM* vm) {
        if (!vm) return;
        jint res = vm->GetEnv((void**)&env, JNI_VERSION_1_6);
        if (res == JNI_EDETACHED) {
            if (vm->AttachCurrentThread(&env, nullptr) == JNI_OK && env) detach = true;
            else env = nullptr;
        } else if (res != JNI_OK) {
            env = nullptr;
        }
    }
    ~JniScope() {
        if (detach && g_jvm) g_jvm->DetachCurrentThread();
    }
};

void notify_progress(uint32_t iter, float pct) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    if (!g_jvm || !g_listener_global || !g_listener_mid) return;
    JniScope scope(g_jvm);
    if (!scope.env) return;
    scope.env->CallVoidMethod(g_listener_global, g_listener_mid, (jint)iter, (jfloat)pct);
    if (scope.env->ExceptionCheck()) scope.env->ExceptionClear();
}

void notify_state(const char* state, const std::string& detail) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    if (!g_jvm || !g_listener_global || !g_listener_state_mid) return;
    JniScope scope(g_jvm);
    if (!scope.env) return;
    jstring js = scope.env->NewStringUTF(state);
    jstring jd = scope.env->NewStringUTF(detail.c_str());
    scope.env->CallVoidMethod(g_listener_global, g_listener_state_mid, js, jd);
    if (scope.env->ExceptionCheck()) scope.env->ExceptionClear();
    if (js) scope.env->DeleteLocalRef(js);
    if (jd) scope.env->DeleteLocalRef(jd);
}

} // namespace

// ---------------------------------------------------------------------------
// Progress callback (runs on the training thread)
// ---------------------------------------------------------------------------

extern "C" void native_progress_callback(ProgressMessage msg, void* user_data) {
    (void)user_data;

    if (msg.tag == 0) {
        g_phase.store(PHASE_PROCESS_STARTED);
        LOGI("Engine: process created (dataset mount + device init starting)");
    } else if (msg.tag == 2) {
        g_phase.store(PHASE_DONE_TRAINING);
        LOGI("Engine: training loop completed");
        return;
    } else if (msg.tag == 1) {
        g_phase.store(PHASE_TRAINING);
        g_last_iter.store(msg.iter);

        uint32_t total = g_total_steps.load();
        float pct = (float)msg.iter / (float)(total > 0 ? total : 1);
        if (pct > 1.0f) pct = 1.0f;
        if (msg.iter % 20 == 0 || msg.iter == total) {
            LOGI("Progress: step %u / %u (%.1f%%)", msg.iter, total, pct * 100.0f);
        }
        notify_progress(msg.iter, pct);
    }

    // ---- cancellation -----------------------------------------------------
    // There is no abort in the engine's C ABI (train_and_save has no cancel
    // token and returns only when the whole run is done), so "cancel" cannot
    // unwind the Rust stack. What it CAN do is stop the work: parking here
    // stops the training thread from ever asking for the next step, so the GPU
    // goes idle and the device stops heating and draining. The cost is that
    // train_and_save never returns and its VRAM/RAM stays held -- the engine is
    // poisoned for the life of the process. Kotlin is told exactly that.
    if (g_cancel_requested.load(std::memory_order_relaxed)) {
        if (!g_engine_poisoned.exchange(true)) {
            LOGE("CANCEL: parking the training thread at step %u. The engine's C ABI "
                 "has no abort, so this run cannot be unwound; its memory stays held "
                 "until the process restarts.", msg.iter);
            notify_state("CANCELLED", "Training stopped at step " + std::to_string(msg.iter) +
                                      "; app restart required before training again");
        }
        for (;;) std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    // ---- thermal / battery duty cycling -----------------------------------
    if (g_paused.load(std::memory_order_relaxed)) {
        LOGW("Duty cycle: PAUSED at step %u (thermal/battery)", msg.iter);
        int waited = 0;
        while (g_paused.load(std::memory_order_relaxed) &&
               !g_cancel_requested.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            waited += 250;
            if (waited % 15000 == 0) LOGW("Duty cycle: still paused (%ds)", waited / 1000);
        }
        LOGI("Duty cycle: resuming after %dms pause", waited);
    }

    int throttle = g_throttle_ms.load(std::memory_order_relaxed);
    if (throttle > 0) {
        int slept = 0;
        while (slept < throttle && !g_cancel_requested.load(std::memory_order_relaxed) &&
               !g_paused.load(std::memory_order_relaxed)) {
            int slice = std::min(100, throttle - slept);
            std::this_thread::sleep_for(std::chrono::milliseconds(slice));
            slept += slice;
        }
    }
}

// ---------------------------------------------------------------------------
// JNI surface
// ---------------------------------------------------------------------------

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeCheckVulkanEngine(
    JNIEnv *env, jobject thiz
) {
    (void)thiz;
    start_log_pump();
    TrainAndSaveFn fn = resolve_train_and_save();
    std::string status = fn
        ? "READY: train_and_save resolved from libbrush_c.so\n"
        : "ERROR: could not resolve train_and_save from libbrush_c.so\n";
    if (g_engine_poisoned.load()) {
        status += "POISONED: a cancelled run parked the engine thread; restart the app before training again\n";
    }
    status += vulkan_report();
    LOGI("Engine diagnostics:\n%s", status.c_str());
    return env->NewStringUTF(status.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeVulkanReport(
    JNIEnv *env, jobject thiz
) {
    (void)thiz;
    std::string report = vulkan_report();
    return env->NewStringUTF(report.c_str());
}

/**
 * Returns:
 *   0 - nothing to cancel (no run in flight)
 *   1 - cancellation requested; the engine thread will park at the next
 *       progress tick and the process must be restarted before training again
 */
JNIEXPORT jint JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeRequestCancel(
    JNIEnv *env, jobject thiz
) {
    (void)env; (void)thiz;
    if (!g_run_active.load()) {
        LOGI("Cancel requested but no training run is active");
        return 0;
    }
    g_cancel_requested.store(true, std::memory_order_relaxed);
    g_paused.store(false, std::memory_order_relaxed);   // so a paused run reaches the cancel check
    LOGW("Cancellation requested; training thread will park at the next progress tick");
    return 1;
}

JNIEXPORT jboolean JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeIsEngineUsable(
    JNIEnv *env, jobject thiz
) {
    (void)env; (void)thiz;
    return g_engine_poisoned.load() ? JNI_FALSE : JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeSetDutyCycle(
    JNIEnv *env, jobject thiz, jint throttleMs, jboolean paused
) {
    (void)env; (void)thiz;
    int ms = throttleMs < 0 ? 0 : (throttleMs > 60000 ? 60000 : (int)throttleMs);
    int prev = g_throttle_ms.exchange(ms);
    bool prev_paused = g_paused.exchange(paused == JNI_TRUE);
    if (prev != ms || prev_paused != (paused == JNI_TRUE)) {
        LOGI("Duty cycle updated: throttle=%dms paused=%d", ms, (int)(paused == JNI_TRUE));
    }
}

/** "" when there is no checkpoint, else "<absolute path>|<iter>|<bytes>". */
JNIEXPORT jstring JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeFindLatestCheckpoint(
    JNIEnv *env, jobject thiz, jstring datasetPath
) {
    (void)thiz;
    if (datasetPath == nullptr) return env->NewStringUTF("");
    const char* c_dataset = env->GetStringUTFChars(datasetPath, nullptr);
    std::string export_dir = std::string(c_dataset) + "/exports";
    env->ReleaseStringUTFChars(datasetPath, c_dataset);

    PlyCandidate cp = find_latest_ply(export_dir, 0);
    if (!cp.valid()) return env->NewStringUTF("");
    std::string s = cp.path + "|" + std::to_string(cp.iter) + "|" + std::to_string(cp.bytes);
    return env->NewStringUTF(s.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeValidateSplat(
    JNIEnv *env, jobject thiz, jstring splatPath, jint maxGaussians
) {
    (void)thiz;
    if (splatPath == nullptr) return env->NewStringUTF("{\"ok\":false,\"message\":\"null path\"}");
    const char* c_path = env->GetStringUTFChars(splatPath, nullptr);
    SplatValidation v = validate_splat(c_path, (long)maxGaussians);
    env->ReleaseStringUTFChars(splatPath, c_path);
    std::string json = validation_json(v);
    return env->NewStringUTF(json.c_str());
}

/**
 * Runs one training job. Returns a JSON result -- the engine's own ABI gives us
 * nothing but 0/1, so everything diagnosable is assembled here.
 */
JNIEXPORT jstring JNICALL
Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeTrainAndSave(
    JNIEnv *env, jobject thiz,
    jstring dataset_path, jstring output_path,
    jint iterations, jint max_resolution,
    jint refine_every, jint export_every, jint max_gaussians,
    jboolean resume,
    jobject progress_listener
) {
    (void)thiz;
    start_log_pump();

    std::string error_code = "NONE";
    std::string message;
    std::string resumed_from;
    bool ok = false;
    int exit_code = -1;
    long produced = 0;
    long long produced_bytes = 0;
    bool capped = false;
    SplatValidation validation;

    auto finish = [&]() -> jstring {
        std::string tail = log_tail_text(40, 4000);
        std::string json = "{";
        json += "\"ok\":" + std::string(ok ? "true" : "false");
        json += ",\"errorCode\":\"" + json_escape(error_code) + "\"";
        json += ",\"message\":\"" + json_escape(message) + "\"";
        json += ",\"exitCode\":" + std::to_string(exit_code);
        json += ",\"phase\":" + std::to_string(g_phase.load());
        json += ",\"lastIter\":" + std::to_string(g_last_iter.load());
        json += ",\"splatCount\":" + std::to_string(produced);
        json += ",\"outputBytes\":" + std::to_string(produced_bytes);
        json += ",\"capped\":" + std::string(capped ? "true" : "false");
        json += ",\"resumedFrom\":\"" + json_escape(resumed_from) + "\"";
        json += ",\"validation\":" + validation_json(validation);
        json += ",\"nativeLog\":\"" + json_escape(tail) + "\"";
        json += "}";
        if (!ok) LOGE("Training result: %s -- %s", error_code.c_str(), message.c_str());
        return env->NewStringUTF(json.c_str());
    };

    if (dataset_path == nullptr || output_path == nullptr) {
        error_code = "BAD_ARGUMENTS";
        message = "Null dataset or output path";
        return finish();
    }
    if (g_engine_poisoned.load()) {
        error_code = "ENGINE_POISONED";
        message = "A previously cancelled run parked the native engine thread. "
                  "The engine's C ABI cannot abort a run, so the app has to be "
                  "restarted before training again.";
        return finish();
    }
    if (g_run_active.exchange(true)) {
        error_code = "ALREADY_RUNNING";
        message = "A training run is already in flight in this process";
        return finish();
    }

    struct RunGuard {
        ~RunGuard() { g_run_active.store(false); }
    } run_guard;

    TrainAndSaveFn train_fn = resolve_train_and_save();
    if (!train_fn) {
        error_code = "ENGINE_MISSING";
        message = "libbrush_c.so did not provide train_and_save. The engine ships "
                  "for arm64-v8a only; on any other ABI on-device training is unavailable.";
        return finish();
    }

    const char* c_dataset = env->GetStringUTFChars(dataset_path, nullptr);
    const char* c_output = env->GetStringUTFChars(output_path, nullptr);
    std::string dataset_dir_str = c_dataset;
    std::string output_str = c_output;
    env->ReleaseStringUTFChars(dataset_path, c_dataset);
    env->ReleaseStringUTFChars(output_path, c_output);

    std::string preflight = dataset_preflight(dataset_dir_str);
    if (!preflight.empty()) {
        error_code = "DATASET_INVALID";
        message = preflight;
        return finish();
    }

    // Reset per-run state.
    g_cancel_requested.store(false, std::memory_order_relaxed);
    g_paused.store(false, std::memory_order_relaxed);
    g_throttle_ms.store(0, std::memory_order_relaxed);
    g_phase.store(PHASE_NOT_STARTED);
    g_last_iter.store(0);
    g_total_steps.store((uint32_t)(iterations > 0 ? iterations : 1));
    clear_log_tail();
    env->GetJavaVM(&g_jvm);

    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        g_listener_global = nullptr;
        g_listener_mid = nullptr;
        g_listener_state_mid = nullptr;
        if (progress_listener != nullptr) {
            g_listener_global = env->NewGlobalRef(progress_listener);
            jclass listener_class = env->GetObjectClass(progress_listener);
            g_listener_mid = env->GetMethodID(listener_class, "onProgress", "(IF)V");
            if (env->ExceptionCheck()) env->ExceptionClear();
            g_listener_state_mid = env->GetMethodID(
                listener_class, "onState", "(Ljava/lang/String;Ljava/lang/String;)V");
            if (env->ExceptionCheck()) env->ExceptionClear();
        }
    }

    std::string export_dir = dataset_dir_str + "/exports";
    std::string init_ply = dataset_dir_str + "/init.ply";
    mkdir(export_dir.c_str(), 0770);

    // ---- warm restart ----------------------------------------------------
    // The C ABI has no "initial point cloud" parameter, but the dataset loader
    // picks `init.ply` out of the dataset VFS ahead of anything else
    // (brush-dataset/src/formats/mod.rs), so staging the newest checkpoint
    // there is the supported way to continue an interrupted run.
    if (resume == JNI_TRUE) {
        PlyCandidate cp = find_latest_ply(export_dir, 0);
        if (cp.valid()) {
            long kept = 0;
            bool staged = decimate_ply(cp.path, init_ply, (long)max_gaussians, &kept);
            if (staged) {
                resumed_from = cp.path;
                LOGI("Warm restart: seeding from checkpoint %s (iter %u, %ld Gaussians) via init.ply",
                     cp.path.c_str(), cp.iter, kept);
                notify_state("RESUMING", "Continuing from checkpoint at step " + std::to_string(cp.iter));
            } else {
                LOGW("Could not stage %s as init.ply; starting from the capture seed cloud",
                     cp.path.c_str());
                unlink(init_ply.c_str());
            }
        } else {
            unlink(init_ply.c_str());
        }
    } else {
        // A stale init.ply or an old run's checkpoints would silently seed this
        // run (and, for a big checkpoint, blow the memory budget doing it).
        unlink(init_ply.c_str());
        remove_plys_in(export_dir);
    }

    // ---- options ----------------------------------------------------------
    uint32_t steps = (uint32_t)(iterations > 0 ? iterations : 1);
    uint32_t effective_export_every = (uint32_t)(export_every > 0 ? export_every : 0);
    if (effective_export_every == 0) {
        effective_export_every = std::max(500u, steps / 5);
    }
    effective_export_every = std::min(effective_export_every, std::max(1u, steps));

    TrainOptions options;
    options.total_train_steps = steps;
    options.refine_every = (uint32_t)(refine_every > 0 ? refine_every : 100);
    options.max_resolution = (uint32_t)(max_resolution > 0 ? max_resolution : 720);
    options.export_every = effective_export_every;
    options.output_path = export_dir.c_str();

    LOGI(">>> Starting on-device 3DGS optimization <<<");
    LOGI("dataset=%s", dataset_dir_str.c_str());
    LOGI("output=%s", output_str.c_str());
    LOGI("steps=%u refineEvery=%u maxRes=%u exportEvery=%u maxGaussians=%d resume=%d",
         options.total_train_steps, options.refine_every, options.max_resolution,
         options.export_every, (int)max_gaussians, (int)(resume == JNI_TRUE));
    LOGI("%s", vulkan_report().c_str());

    long long pre_output_bytes = file_size(output_str);
    time_t run_start = time(nullptr) - 2;   // mtime granularity slack

    TrainExitCode result = train_fn(dataset_dir_str.c_str(), &options,
                                    native_progress_callback, nullptr);
    exit_code = (int)result;
    int phase = g_phase.load();
    uint32_t last_iter = g_last_iter.load();
    LOGI("train_and_save returned %d (phase=%d lastIter=%u)", exit_code, phase, last_iter);

    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        if (g_listener_global) {
            env->DeleteGlobalRef(g_listener_global);
            g_listener_global = nullptr;
            g_listener_mid = nullptr;
            g_listener_state_mid = nullptr;
        }
    }

    std::string tail = log_tail_text(12, 1500);

    if (result != TrainSuccess) {
        // Attribute the bare exit code to the phase it died in. brush-c matches
        // its internal error as `Err(_)` and returns 1, so the phase plus any
        // panic text captured off stderr is all the detail that exists.
        switch (phase) {
            case PHASE_NOT_STARTED:
                error_code = "ENGINE_INIT_FAILED";
                message = "The engine failed before it created a process -- it never mounted "
                          "the dataset or reached GPU init. Typically a wgpu/Vulkan adapter "
                          "failure or an unsupported compute feature.";
                break;
            case PHASE_PROCESS_STARTED:
                error_code = "DATASET_LOAD_FAILED";
                message = "The engine started but failed before the first training step -- "
                          "dataset parsing, image decode, GPU device creation or the initial "
                          "point cloud. On a 4GB device this is usually memory.";
                break;
            default:
                error_code = "TRAINING_FAILED";
                message = "Training aborted at step " + std::to_string(last_iter) + " of " +
                          std::to_string(steps) + ".";
                break;
        }
        if (!tail.empty()) message += "\nEngine output:\n" + tail;
        else message += "\n(The engine printed nothing; it reports internal errors only as "
                        "exit code 1 and installs no logger.)";
        return finish();
    }

    if (g_cancel_requested.load()) {
        error_code = "CANCELLED";
        message = "Run cancelled at step " + std::to_string(last_iter);
        return finish();
    }

    // ---- pick up this run's checkpoint ------------------------------------
    // Only a PLY this run wrote counts. Accepting an older one is how a failed
    // run ends up presented as a fresh result.
    PlyCandidate out_ply = find_latest_ply(export_dir, run_start);
    if (!out_ply.valid()) {
        error_code = "NO_CHECKPOINT";
        message = "The engine reported success but wrote no .ply into " + export_dir +
                  " during this run.";
        if (!tail.empty()) message += "\nEngine output:\n" + tail;
        return finish();
    }
    LOGI("Using checkpoint %s (iter %u, %lld bytes)",
         out_ply.path.c_str(), out_ply.iter, out_ply.bytes);

    // ---- convert into a staging file, validate, then publish --------------
    // Never write the destination until the bytes are known good: the target
    // path may already hold a previously presented model.
    std::string staging = output_str + ".part";
    unlink(staging.c_str());

    bool want_ply = output_str.size() > 4 &&
                    output_str.substr(output_str.size() - 4) == ".ply";
    if (want_ply) {
        if (!copy_file(out_ply.path, staging)) {
            error_code = "CONVERSION_FAILED";
            message = "Could not copy the trained PLY to " + staging;
            unlink(staging.c_str());
            return finish();
        }
        produced_bytes = file_size(staging);
        validation.ok = produced_bytes > 0;
        if (!validation.ok) {
            error_code = "VALIDATION_FAILED";
            message = "Copied PLY is empty";
            unlink(staging.c_str());
            return finish();
        }
    } else {
        ConvertResult conv = convert_ply_to_splat(out_ply.path, staging, (long)max_gaussians);
        capped = conv.capped;
        if (!conv.ok) {
            error_code = "CONVERSION_FAILED";
            message = conv.error;
            unlink(staging.c_str());
            return finish();
        }
        produced = conv.written;

        validation = validate_splat(staging, (long)max_gaussians);
        produced_bytes = validation.bytes;
        if (!validation.ok) {
            error_code = "VALIDATION_FAILED";
            message = "The trained model failed validation and was discarded: " + validation.error;
            unlink(staging.c_str());
            return finish();
        }
        LOGI("Validation OK: %ld Gaussians, %lld bytes, %ld opaque",
             validation.count, validation.bytes, validation.opaque);
    }

    if (rename(staging.c_str(), output_str.c_str()) != 0) {
        error_code = "PUBLISH_FAILED";
        message = std::string("Could not move the validated model into place: ") + strerror(errno);
        unlink(staging.c_str());
        return finish();
    }

    produced_bytes = file_size(output_str);
    if (pre_output_bytes >= 0) {
        LOGI("Replaced pre-existing output (%lld bytes) with the trained model (%lld bytes)",
             pre_output_bytes, produced_bytes);
    }

    ok = true;
    message = "Trained " + std::to_string(produced > 0 ? produced : validation.count) +
              " Gaussians in " + std::to_string(last_iter) + " steps";
    return finish();
}

} // extern "C"
