/*
 * Mobile 3D Gaussian Splatting viewer - WebGL2.
 *
 * Model format: 32 bytes per splat
 *   [ 0..11]  position   float32 x3
 *   [12..23]  scale      float32 x3   (linear, world units)
 *   [24..27]  colour     uint8   RGBA
 *   [28..31]  rotation   uint8   quaternion, SCALAR FIRST (w,x,y,z), q = r*128 + 128
 *
 * Rendering strategy
 *   - The whole model lives in one RGBA32UI texture (2 texels per splat, a byte
 *     for byte copy of the file - no CPU repacking).
 *   - The only per-instance vertex attribute is a uint32 index into that texture.
 *     Re-ordering for depth therefore uploads 4 bytes per splat instead of 32,
 *     and the 8-57 MB model buffer is uploaded exactly once.
 *   - Culling + depth sorting run as an incremental job spread over frames with a
 *     per-frame time budget, so a re-sort never blocks a frame.
 *   - The render list is stored far -> near. "Draw only the nearest N" is then a
 *     byte offset into the index buffer: LOD costs zero uploads and zero CPU.
 */
'use strict';

(function () {

const SPLAT_BYTES = 32;
const WORDS_PER_SPLAT = 8;
const BUCKETS = 65536;
const SORT_CHUNK = 32768;
const INVALID_BUCKET = 0xFFFF;

/* ------------------------------------------------------------------ utils */

const $ = (id) => document.getElementById(id);

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

function fmt(n) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function metersLabel(m) {
    if (m >= 1.0) return m.toFixed(2) + ' m';
    return (m * 100).toFixed(1) + ' cm';
}

/* ------------------------------------------------------------------- UI */

const UI = {
    toastTimer: 0,
    hintTimer: 0,

    setLoading(text, sub) {
        const el = $('loading');
        el.classList.remove('gone');
        el.style.display = '';
        $('loading-text').innerHTML = text;
        $('loading-sub').textContent = sub || '';
    },
    setProgress(frac) {
        const ring = $('load-ring');
        const bar = $('load-bar');
        if (frac === null || frac === undefined || !isFinite(frac)) {
            ring.classList.add('indeterminate');
            return;
        }
        ring.classList.remove('indeterminate');
        const C = 2 * Math.PI * 24;
        bar.style.strokeDashoffset = String(C * (1 - clamp(frac, 0, 1)));
    },
    hideLoading() {
        const el = $('loading');
        el.classList.add('gone');
        setTimeout(() => { if (el.classList.contains('gone')) el.style.display = 'none'; }, 420);
    },
    toast(msg, ms) {
        const el = $('toast') || (() => {
            const d = document.createElement('div');
            d.id = 'toast';
            document.body.appendChild(d);
            return d;
        })();
        el.textContent = msg;
        // force style flush so the transition always runs
        void el.offsetWidth;
        el.classList.add('show');
        clearTimeout(UI.toastTimer);
        UI.toastTimer = setTimeout(() => el.classList.remove('show'), ms || 2200);
    },
    hint(msg) {
        const chip = $('chip-hint');
        if (!msg) { chip.classList.remove('show'); return; }
        $('hint-text').textContent = msg;
        chip.classList.add('show');
    },
    fatal(msg) {
        $('fatal-inner').textContent = msg;
        $('fatal').classList.add('show');
        UI.hideLoading();
    }
};

/* --------------------------------------------------------------- shaders */

const VS = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec2 a_quad;
layout(location = 1) in uint a_index;

uniform highp usampler2D u_splats;
uniform int  u_texWidth;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport;
uniform vec2 u_focal;

out vec4 v_color;
out vec2 v_coord;

mat3 quatToMat(vec4 q) {
    float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    float xx = q.x * x2,  xy = q.x * y2,  xz = q.x * z2;
    float yy = q.y * y2,  yz = q.y * z2,  zz = q.z * z2;
    float wx = q.w * x2,  wy = q.w * y2,  wz = q.w * z2;
    return mat3(
        1.0 - (yy + zz), xy + wz, xz - wy,
        xy - wz, 1.0 - (xx + zz), yz + wx,
        xz + wy, yz - wx, 1.0 - (xx + yy)
    );
}

void main() {
    // Two texels hold one 32-byte splat record, laid out exactly as on disk:
    //   texel0 = px, py, pz, sx     texel1 = sy, sz, rgba, quat
    int base = int(a_index) << 1;
    ivec2 t0 = ivec2(base % u_texWidth, base / u_texWidth);
    int b1 = base + 1;
    ivec2 t1 = ivec2(b1 % u_texWidth, b1 / u_texWidth);
    uvec4 w0 = texelFetch(u_splats, t0, 0);
    uvec4 w1 = texelFetch(u_splats, t1, 0);

    vec3 center = vec3(uintBitsToFloat(w0.x), uintBitsToFloat(w0.y), uintBitsToFloat(w0.z));
    vec3 scale  = vec3(uintBitsToFloat(w0.w), uintBitsToFloat(w1.x), uintBitsToFloat(w1.y));
    vec4 rgba   = vec4((uvec4(w1.z) >> uvec4(0u, 8u, 16u, 24u)) & 255u) / 255.0;
    vec4 qb     = vec4((uvec4(w1.w) >> uvec4(0u, 8u, 16u, 24u)) & 255u);

    v_color = rgba;
    v_coord = a_quad;

    vec4 cam = u_view * vec4(center, 1.0);
    vec4 clipPos = u_proj * cam;

    // Safety cull (the CPU render list already culls, this catches stale lists).
    float lim = 1.35 * clipPos.w;
    if (clipPos.w <= 0.0 || abs(clipPos.x) > lim || abs(clipPos.y) > lim) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    // The .splat writer packs q = r * 128 + 128 in (w, x, y, z) order, so the
    // exact inverse is (q / 128) - 1 and the scalar part is component .x --
    // quatToMat() expects the scalar in .w.
    vec4 rq = normalize(qb / 128.0 - 1.0);
    mat3 R = quatToMat(vec4(rq.yzw, rq.x));
    mat3 S = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    mat3 M = R * S;
    mat3 cov3d = M * transpose(M);          // 3D covariance in world space

    // Project the 3D covariance into 2D screen space: Sigma2D = J W Sigma W^T J^T.
    // (mat3 ctor is column-major, so this literal is J^T; combined with
    // transpose(mat3(u_view)) below the quadratic form works out to J W S W^T J^T.)
    mat3 J = mat3(
        u_focal.x / cam.z, 0.0, -(u_focal.x * cam.x) / (cam.z * cam.z),
        0.0, u_focal.y / cam.z, -(u_focal.y * cam.y) / (cam.z * cam.z),
        0.0, 0.0, 0.0
    );
    mat3 T = transpose(mat3(u_view)) * J;
    mat3 cov2d = transpose(T) * cov3d * T;

    // Low-pass filter so sub-pixel splats stay at least one pixel wide.
    float a = cov2d[0][0] + 0.3;
    float b = cov2d[0][1];
    float d = cov2d[1][1] + 0.3;

    // Eigen-decomposition of the 2x2 covariance -> billboard axes (pixels).
    float det = a * d - b * b;
    if (det <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
    float mid = 0.5 * (a + d);
    float rad = sqrt(max(0.0, mid * mid - det));
    float lambda1 = mid + rad;
    float lambda2 = max(mid - rad, 0.1);
    vec2 dir = normalize(vec2(b, lambda1 - a));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * dir;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(dir.y, -dir.x);

    // a_quad spans [-2, 2], so the quad covers ~2.83 sigma.
    vec2 ndcCenter = clipPos.xy / clipPos.w;
    vec2 offsetPx = a_quad.x * majorAxis + a_quad.y * minorAxis;
    gl_Position = vec4(
        ndcCenter + offsetPx * 2.0 / u_viewport,
        clipPos.z / clipPos.w,
        1.0
    );
}`;

const FS = `#version 300 es
precision highp float;

in vec4 v_color;
in vec2 v_coord;
out vec4 fragColor;

void main() {
    // v_coord spans [-2, 2] and the quad is sized to ~2.83 sigma, so the
    // Gaussian evaluates to exp(-r^2) in quad space.
    float A = dot(v_coord, v_coord);
    if (A > 4.0) discard;

    float alpha = exp(-A) * v_color.a;
    if (alpha < 0.004) discard;

    fragColor = vec4(v_color.rgb * alpha, alpha);
}`;

/* ------------------------------------------------------------- the viewer */

class GaussianSplatViewer {
    constructor() {
        this.canvas = $('gl-canvas');
        this.ok = false;

        this.gl = this.canvas.getContext('webgl2', {
            antialias: false, alpha: false, depth: false, stencil: false,
            premultipliedAlpha: true, preserveDrawingBuffer: false,
            powerPreference: 'high-performance', desynchronized: true
        });
        if (!this.gl) {
            UI.fatal('WebGL2 is not available on this device, so the 3D viewer cannot run.');
            return;
        }

        this.caps = this.readDeviceCaps();

        /* ---- model state ---- */
        this.splatBuffer = null;     // ArrayBuffer, authoritative CPU copy
        this.f32 = null;             // Float32Array view of it
        this.u8 = null;              // Uint8Array view of it
        this.splatCount = 0;
        this.sourceUrl = null;
        this.edited = false;

        /* ---- render list ---- */
        this.order = null;           // Uint32Array, far -> near
        this.bucketOf = null;        // Uint16Array
        this.counts = new Uint32Array(BUCKETS);
        this.visibleCount = 0;       // entries currently valid in the GPU index buffer
        this.maxOrder = 0;
        this.orderValid = false;

        /* ---- incremental sort job ---- */
        this.job = { phase: 'idle', cursor: 0 };
        this.lastSortAt = -1e9;
        this.lastSortFwd = null;
        this.lastSortEye = null;
        this.sortDirty = true;

        /* ---- camera ---- */
        this.camera = {
            target: [0, 0, 0],
            radius: 3.0,
            theta: 0.0,
            phi: Math.PI / 3.2,
            fov: 50 * Math.PI / 180,
            near: 0.05,
            far: 200.0
        };
        this.home = null;
        this.vel = { theta: 0, phi: 0, radius: 0, tx: 0, ty: 0, tz: 0 };
        this.scene = { center: [0, 0, 0], radius: 1, minY: 0, maxY: 0, medScale: 0.01 };

        /* ---- timing / LOD ---- */
        this.frameMs = 16.7;
        this.fps = 0;
        this.lastFrame = 0;
        this.budget = 1;             // eased splat budget actually drawn
        this.autoBudget = this.caps.startBudget;
        this.lastTune = 0;
        this.tuneFrames = 0;
        this.tuneAccum = 0;
        this.movingUntil = 0;
        this.lastHudAt = 0;

        /* ---- tools ---- */
        this.measuring = false;
        this.measurePoints = [];
        this.saving = false;

        this.themeDark = false;

        try {
            this.initGL();
        } catch (e) {
            UI.fatal(String(e && e.message ? e.message : e));
            return;
        }

        this.initEvents();
        this.resize();
        this.applyTheme(false);
        this.ok = true;

        this.lastFrame = performance.now();
        this.rafId = requestAnimationFrame((t) => this.renderLoop(t));
    }

    /* ------------------------------------------------------ device caps */

    readDeviceCaps() {
        let totalMem = 0, lowRam = false;
        try {
            if (window.AndroidBridge && window.AndroidBridge.getDeviceInfo) {
                const info = JSON.parse(window.AndroidBridge.getDeviceInfo());
                totalMem = info.totalMemMb | 0;
                lowRam = !!info.lowRam;
            }
        } catch (e) { /* fall through to heuristics */ }
        if (!totalMem && navigator.deviceMemory) totalMem = navigator.deviceMemory * 1024;
        if (!totalMem) totalMem = 3072;

        // Hard ceiling on splats kept in RAM + VRAM. 32 bytes each on the CPU and
        // 32 bytes each in the texture, so 1 M splats ~= 64 MB total.
        let maxSplats;
        if (lowRam || totalMem < 2600) maxSplats = 700000;
        else if (totalMem < 4200) maxSplats = 1400000;
        else if (totalMem < 6500) maxSplats = 2400000;
        else maxSplats = 3500000;

        const startBudget = Math.min(maxSplats, lowRam || totalMem < 2600 ? 250000 : 900000);
        return { totalMem, lowRam, maxSplats, startBudget };
    }

    /* ------------------------------------------------------------- GL */

    initGL() {
        const gl = this.gl;

        this.program = this.buildProgram(VS, FS);
        gl.useProgram(this.program);
        this.uni = {
            view: gl.getUniformLocation(this.program, 'u_view'),
            proj: gl.getUniformLocation(this.program, 'u_proj'),
            viewport: gl.getUniformLocation(this.program, 'u_viewport'),
            focal: gl.getUniformLocation(this.program, 'u_focal'),
            splats: gl.getUniformLocation(this.program, 'u_splats'),
            texWidth: gl.getUniformLocation(this.program, 'u_texWidth')
        };
        gl.uniform1i(this.uni.splats, 0);

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        // Quad spans [-2, 2] so the projected billboard covers ~2.83 sigma.
        this.quadVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(0, 0);

        this.orderVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.orderVBO);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
        gl.vertexAttribDivisor(1, 1);

        gl.bindVertexArray(null);

        this.splatTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.splatTex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);

        this.maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 2048;
        this.texWidth = Math.min(2048, this.maxTexSize);
        // 2 texels per splat.
        this.texCapacity = Math.floor(this.texWidth * this.maxTexSize / 2);

        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }

    buildProgram(vsSrc, fsSrc) {
        const gl = this.gl;
        const vs = this.compileShader(gl.VERTEX_SHADER, vsSrc);
        const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSrc);
        const p = gl.createProgram();
        gl.attachShader(p, vs);
        gl.attachShader(p, fs);
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            const log = gl.getProgramInfoLog(p);
            console.error('Program link failed: ' + log);
            throw new Error('Shader program link failed:\n' + log);
        }
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return p;
    }

    compileShader(type, src) {
        const gl = this.gl;
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(s);
            const kind = (type === gl.VERTEX_SHADER) ? 'vertex' : 'fragment';
            console.error(kind + ' shader compile failed: ' + log);
            gl.deleteShader(s);
            throw new Error(kind + ' shader compile failed:\n' + log);
        }
        return s;
    }

    /* ----------------------------------------------------------- loading */

    loadSplat(url, keepCamera) {
        this.sourceUrl = url;
        this.keepCameraOnLoad = !!keepCamera;
        UI.setLoading('Loading 3D scene&hellip;', '');
        UI.setProgress(null);

        const xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.responseType = 'arraybuffer';
        xhr.onprogress = (e) => {
            const mb = (b) => (b / 1048576).toFixed(1);
            if (e.lengthComputable && e.total > 0) {
                UI.setProgress(e.loaded / e.total);
                $('loading-sub').textContent = mb(e.loaded) + ' / ' + mb(e.total) + ' MB';
            } else {
                $('loading-sub').textContent = mb(e.loaded) + ' MB';
            }
        };
        xhr.onload = () => {
            if ((xhr.status === 200 || xhr.status === 0) && xhr.response && xhr.response.byteLength >= SPLAT_BYTES) {
                UI.setProgress(1);
                UI.setLoading('Preparing splats&hellip;', '');
                // Let the progress ring paint before the synchronous upload.
                setTimeout(() => {
                    try {
                        this.setSplatData(xhr.response);
                    } catch (e) {
                        console.error(e);
                        UI.fatal('Could not prepare the model: ' + (e && e.message ? e.message : e));
                    }
                }, 30);
            } else {
                UI.setLoading('Could not load the model file.',
                    'HTTP ' + xhr.status + (xhr.response ? ' - ' + xhr.response.byteLength + ' bytes' : ''));
                UI.setProgress(0);
            }
        };
        xhr.onerror = () => {
            UI.setLoading('Could not reach the model file.', String(url));
            UI.setProgress(0);
        };
        try {
            xhr.send();
        } catch (e) {
            UI.setLoading('Could not start the download.', String(e));
        }
    }

    /**
     * Install a new splat buffer: cap it to what this device can hold, upload it
     * to the data texture, (re)allocate the sort scratch and frame the camera.
     */
    setSplatData(arrayBuffer) {
        const gl = this.gl;
        let count = Math.floor(arrayBuffer.byteLength / SPLAT_BYTES);
        if (count <= 0) throw new Error('The model file contains no splats.');

        const cap = Math.min(this.caps.maxSplats, this.texCapacity);
        let decimated = 0;
        if (count > cap) {
            decimated = count;
            arrayBuffer = GaussianSplatViewer.decimate(arrayBuffer, count, cap);
            count = Math.floor(arrayBuffer.byteLength / SPLAT_BYTES);
        }

        this.splatBuffer = arrayBuffer;
        this.splatCount = count;
        this.u8 = new Uint8Array(arrayBuffer);
        this.f32 = new Float32Array(arrayBuffer, 0, count * WORDS_PER_SPLAT);

        this.computeSceneBounds();
        this.uploadSplatTexture();

        // Sort scratch. bucketOf is Uint16 (one per splat); order is capped so a
        // re-upload never exceeds a few MB.
        this.bucketOf = new Uint16Array(count);
        this.maxOrder = Math.min(count, this.caps.maxSplats);
        this.order = new Uint32Array(this.maxOrder);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.orderVBO);
        gl.bufferData(gl.ARRAY_BUFFER, this.maxOrder * 4, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        this.visibleCount = 0;
        this.orderValid = false;
        this.job.phase = 'idle';
        this.lastSortFwd = null;
        this.lastSortEye = null;
        this.sortDirty = true;
        this.autoBudget = Math.min(this.caps.startBudget, count);
        this.budget = this.autoBudget;
        this.measurePoints = [];
        this.updateMeasureOverlay();

        if (!this.keepCameraOnLoad || !this.home) this.frameCamera();
        this.keepCameraOnLoad = false;

        $('stat-total').textContent = fmt(count);
        $('btn-restore').classList.toggle('hidden', !this.edited);

        // Build the first render list synchronously so nothing pops in.
        this.updateFrameConstants();
        this.beginSortJob(this.lastBasis);
        this.pumpSortJob(1e9);

        UI.hideLoading();
        if (decimated) {
            UI.toast('Model reduced to ' + fmt(count) + ' of ' + fmt(decimated) +
                ' splats to fit this device', 4200);
        }
    }

    /** Uniform stride decimation, preserving the 32-byte record layout. */
    static decimate(buffer, count, target) {
        const src = new Uint32Array(buffer, 0, count * WORDS_PER_SPLAT);
        const out = new Uint32Array(target * WORDS_PER_SPLAT);
        const step = count / target;
        for (let k = 0; k < target; k++) {
            const i = Math.min(count - 1, Math.floor(k * step));
            const s = i * WORDS_PER_SPLAT, d = k * WORDS_PER_SPLAT;
            out[d] = src[s]; out[d + 1] = src[s + 1]; out[d + 2] = src[s + 2]; out[d + 3] = src[s + 3];
            out[d + 4] = src[s + 4]; out[d + 5] = src[s + 5]; out[d + 6] = src[s + 6]; out[d + 7] = src[s + 7];
        }
        return out.buffer;
    }

    uploadSplatTexture() {
        const gl = this.gl;
        const texels = this.splatCount * 2;
        const w = this.texWidth;
        const h = Math.max(1, Math.ceil(texels / w));
        if (h > this.maxTexSize) throw new Error('Model is too large for this GPU.');

        gl.bindTexture(gl.TEXTURE_2D, this.splatTex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, w, h, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, null);

        // Views over the model buffer - no copy, no padded staging allocation.
        const words = new Uint32Array(this.splatBuffer, 0, this.splatCount * WORDS_PER_SPLAT);
        const fullRows = Math.floor(texels / w);
        if (fullRows > 0) {
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, fullRows, gl.RGBA_INTEGER, gl.UNSIGNED_INT,
                words.subarray(0, fullRows * w * 4));
        }
        const rem = texels - fullRows * w;
        if (rem > 0) {
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, fullRows, rem, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT,
                words.subarray(fullRows * w * 4, fullRows * w * 4 + rem * 4));
        }
        gl.bindTexture(gl.TEXTURE_2D, null);

        const err = gl.getError();
        if (err !== gl.NO_ERROR) console.warn('GL error after texture upload: 0x' + err.toString(16));
        this.texHeight = h;
    }

    /** Robust scene bounds (percentile based, so a few floaters cannot skew it). */
    computeSceneBounds() {
        const f = this.f32, n = this.splatCount;
        let cx = 0, cy = 0, cz = 0;
        let minY = Infinity, maxY = -Infinity;
        const step = Math.max(1, Math.floor(n / 200000));
        let m = 0;
        for (let i = 0; i < n; i += step) {
            const o = i * WORDS_PER_SPLAT;
            cx += f[o]; cy += f[o + 1]; cz += f[o + 2];
            const y = f[o + 1];
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            m++;
        }
        if (m === 0) m = 1;
        cx /= m; cy /= m; cz /= m;

        // 92nd percentile radius via a coarse histogram of squared distance.
        let maxR2 = 0;
        const d2 = new Float32Array(Math.ceil(n / step));
        let k = 0;
        for (let i = 0; i < n; i += step) {
            const o = i * WORDS_PER_SPLAT;
            const dx = f[o] - cx, dy = f[o + 1] - cy, dz = f[o + 2] - cz;
            const r2 = dx * dx + dy * dy + dz * dz;
            d2[k++] = r2;
            if (r2 > maxR2) maxR2 = r2;
        }
        const HB = 512, hist = new Uint32Array(HB);
        const hs = maxR2 > 0 ? (HB - 1) / maxR2 : 0;
        for (let i = 0; i < k; i++) hist[(d2[i] * hs) | 0]++;
        let acc = 0, cut = HB - 1;
        const want = k * 0.92;
        for (let b = 0; b < HB; b++) { acc += hist[b]; if (acc >= want) { cut = b; break; } }
        const r92 = Math.sqrt((cut + 1) / (hs || 1));

        // Typical splat size, used for scene-relative cleaning thresholds.
        let sAcc = 0, sN = 0;
        for (let i = 0; i < n; i += step) {
            const o = i * WORDS_PER_SPLAT;
            sAcc += Math.abs(f[o + 3]) + Math.abs(f[o + 4]) + Math.abs(f[o + 5]);
            sN += 3;
        }

        this.scene.center = [cx, cy, cz];
        this.scene.radius = Math.max(1e-3, isFinite(r92) ? r92 : 1);
        this.scene.maxRadius = Math.max(this.scene.radius, Math.sqrt(maxR2) || this.scene.radius);
        this.scene.minY = minY; this.scene.maxY = maxY;
        this.scene.medScale = sN ? Math.max(1e-5, sAcc / sN) : 0.01;
    }

    frameCamera() {
        const c = this.camera;
        c.target = this.scene.center.slice();
        c.radius = clamp(this.scene.radius / Math.sin(c.fov * 0.5) * 0.95,
            this.scene.radius * 0.15, this.scene.radius * 12);
        c.theta = 0.0;
        c.phi = Math.PI / 3.2;
        c.near = Math.max(0.01, this.scene.radius * 0.005);
        c.far = this.scene.maxRadius * 24 + c.radius * 4;
        this.home = { target: c.target.slice(), radius: c.radius, theta: c.theta, phi: c.phi };
        this.vel = { theta: 0, phi: 0, radius: 0, tx: 0, ty: 0, tz: 0 };
        this.sortDirty = true;
    }

    resetCamera() {
        if (!this.home) return;
        const c = this.camera;
        c.target = this.home.target.slice();
        c.radius = this.home.radius;
        c.theta = this.home.theta;
        c.phi = this.home.phi;
        this.vel = { theta: 0, phi: 0, radius: 0, tx: 0, ty: 0, tz: 0 };
        this.sortDirty = true;
        this.markMoving();
    }

    /* ------------------------------------------------------- camera math */

    cameraBasis() {
        const c = this.camera;
        const sp = Math.sin(c.phi), cp = Math.cos(c.phi);
        const st = Math.sin(c.theta), ct = Math.cos(c.theta);
        const eye = [
            c.target[0] + c.radius * sp * st,
            c.target[1] + c.radius * cp,
            c.target[2] + c.radius * sp * ct
        ];
        let fx = c.target[0] - eye[0], fy = c.target[1] - eye[1], fz = c.target[2] - eye[2];
        const fl = Math.hypot(fx, fy, fz) || 1;
        fx /= fl; fy /= fl; fz /= fl;
        // right = normalize(fwd x up)
        let rx = fy * 0 - fz * 1, ry = fz * 0 - fx * 0, rz = fx * 1 - fy * 0;
        const rl = Math.hypot(rx, ry, rz) || 1;
        rx /= rl; ry /= rl; rz /= rl;
        // up = right x fwd
        const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
        return { eye, fwd: [fx, fy, fz], right: [rx, ry, rz], up: [ux, uy, uz] };
    }

    viewMatrix(b) {
        const e = b.eye, r = b.right, u = b.up, f = b.fwd;
        // z axis of the view basis points backwards (= -forward)
        const zx = -f[0], zy = -f[1], zz = -f[2];
        return new Float32Array([
            r[0], u[0], zx, 0,
            r[1], u[1], zy, 0,
            r[2], u[2], zz, 0,
            -(r[0] * e[0] + r[1] * e[1] + r[2] * e[2]),
            -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]),
            -(zx * e[0] + zy * e[1] + zz * e[2]), 1
        ]);
    }

    projMatrix(aspect) {
        const c = this.camera;
        const f = 1.0 / Math.tan(c.fov / 2);
        const nf = 1 / (c.near - c.far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (c.far + c.near) * nf, -1,
            0, 0, 2 * c.far * c.near * nf, 0
        ]);
    }

    /** Refresh the cached camera basis and projection factors. */
    updateFrameConstants() {
        this.lastBasis = this.cameraBasis();
        const aspect = this.canvas.width / Math.max(1, this.canvas.height);
        const f = 1.0 / Math.tan(this.camera.fov / 2);
        this.p00 = f / aspect;
        this.p11 = f;
        return this.lastBasis;
    }

    /** World point -> CSS pixel coordinates, or null when behind the camera. */
    projectPoint(p) {
        const b = this.lastBasis;
        if (!b) return null;
        const dx = p[0] - b.eye[0], dy = p[1] - b.eye[1], dz = p[2] - b.eye[2];
        const z = dx * b.fwd[0] + dy * b.fwd[1] + dz * b.fwd[2];
        if (z <= this.camera.near) return null;
        const cx = dx * b.right[0] + dy * b.right[1] + dz * b.right[2];
        const cy = dx * b.up[0] + dy * b.up[1] + dz * b.up[2];
        const w = this.cssW, h = this.cssH;
        const ndcX = (this.p00 * cx) / z;
        const ndcY = (this.p11 * cy) / z;
        return [(ndcX * 0.5 + 0.5) * w, (0.5 - ndcY * 0.5) * h];
    }

    /* ---------------------------------------------- incremental sort job */

    requestSort(force) {
        this.sortDirty = true;
        if (force) this.lastSortAt = -1e9;
    }

    /** Decide whether a new cull+sort job should start this frame. */
    maybeStartSortJob(now, moving) {
        if (this.job.phase !== 'idle') return;
        if (!this.splatCount) return;

        const b = this.lastBasis;
        if (!this.orderValid) { this.beginSortJob(b); return; }
        if (!this.sortDirty) return;

        // How stale is the current order?
        let stale = 1;
        if (this.lastSortFwd) {
            const d = b.fwd[0] * this.lastSortFwd[0] + b.fwd[1] * this.lastSortFwd[1] + b.fwd[2] * this.lastSortFwd[2];
            const moveSq = (b.eye[0] - this.lastSortEye[0]) ** 2 +
                           (b.eye[1] - this.lastSortEye[1]) ** 2 +
                           (b.eye[2] - this.lastSortEye[2]) ** 2;
            const rel = Math.sqrt(moveSq) / (this.scene.radius || 1);
            stale = Math.max(1 - d, rel * 0.5);
            if (stale < 0.00025) { this.sortDirty = false; return; }
        }

        const since = now - this.lastSortAt;
        // Settled: re-sort as soon as the camera stops. Moving: re-sort lazily,
        // and only once the order is meaningfully wrong.
        if (!moving) {
            if (since > 90) this.beginSortJob(b);
        } else if (since > 320 && stale > 0.012) {
            this.beginSortJob(b);
        }
    }

    beginSortJob(b) {
        const job = this.job;
        job.eye = b.eye.slice();
        job.fwd = b.fwd.slice();
        job.right = b.right.slice();
        job.up = b.up.slice();
        job.p00 = this.p00;
        job.p11 = this.p11;
        job.near = this.camera.near;

        // Exact depth range from the scene bounding box projected onto fwd, so
        // the counting sort needs no min/max pre-pass.
        const c = this.scene.center, r = this.scene.maxRadius;
        const dc = (c[0] - b.eye[0]) * b.fwd[0] + (c[1] - b.eye[1]) * b.fwd[1] + (c[2] - b.eye[2]) * b.fwd[2];
        job.minD = Math.max(job.near, dc - r * 1.05);
        job.maxD = Math.max(job.minD + 1e-4, dc + r * 1.05);
        job.scale = (BUCKETS - 1) / (job.maxD - job.minD);

        this.counts.fill(0);
        job.cursor = 0;
        job.phase = 'count';
        job.startedAt = performance.now();
    }

    /** Run the pending job for at most `budgetMs`. Returns true if it finished. */
    pumpSortJob(budgetMs) {
        const job = this.job;
        if (job.phase === 'idle') return false;
        const t0 = performance.now();
        const n = this.splatCount;
        const f = this.f32, bucketOf = this.bucketOf, counts = this.counts;

        while (job.phase !== 'idle') {
            if (job.phase === 'count') {
                const ex = job.eye[0], ey = job.eye[1], ez = job.eye[2];
                const fx = job.fwd[0], fy = job.fwd[1], fz = job.fwd[2];
                const rx = job.right[0], ry = job.right[1], rz = job.right[2];
                const ux = job.up[0], uy = job.up[1], uz = job.up[2];
                const p00 = job.p00, p11 = job.p11, near = job.near;
                const minD = job.minD, sc = job.scale;

                let i = job.cursor;
                const end = Math.min(n, i + SORT_CHUNK);
                for (; i < end; i++) {
                    const o = i * WORDS_PER_SPLAT;
                    const dx = f[o] - ex, dy = f[o + 1] - ey, dz = f[o + 2] - ez;
                    const z = dx * fx + dy * fy + dz * fz;
                    if (!(z > near)) { bucketOf[i] = INVALID_BUCKET; continue; }

                    // Conservative screen extent of this splat (3 sigma).
                    let sx = f[o + 3]; if (sx < 0) sx = -sx;
                    let sy = f[o + 4]; if (sy < 0) sy = -sy;
                    let sz2 = f[o + 5]; if (sz2 < 0) sz2 = -sz2;
                    let rad = sx > sy ? sx : sy; if (sz2 > rad) rad = sz2;
                    rad *= 3.0;

                    // |p00 * camX / z| > 1 + p00 * rad / z  <=>  |p00*camX| > z + p00*rad
                    const cX = dx * rx + dy * ry + dz * rz;
                    const lx = p00 * cX;
                    if (lx > z + p00 * rad || -lx > z + p00 * rad) { bucketOf[i] = INVALID_BUCKET; continue; }
                    const cY = dx * ux + dy * uy + dz * uz;
                    const ly = p11 * cY;
                    if (ly > z + p11 * rad || -ly > z + p11 * rad) { bucketOf[i] = INVALID_BUCKET; continue; }

                    let bk = ((z - minD) * sc) | 0;
                    if (bk < 0) bk = 0; else if (bk > BUCKETS - 2) bk = BUCKETS - 2;
                    bucketOf[i] = bk;
                    counts[bk]++;
                }
                job.cursor = i;
                if (i >= n) { job.phase = 'prefix'; }

            } else if (job.phase === 'prefix') {
                // Walk near -> far until the budget is full; that bucket is the cut.
                const cap = this.maxOrder;
                let acc = 0, cut = BUCKETS - 2, limit = 0;
                for (let b = 0; b < BUCKETS - 1; b++) {
                    const c = counts[b];
                    if (acc + c >= cap) { cut = b; limit = cap - acc; acc = cap; break; }
                    acc += c;
                }
                if (acc < cap) { limit = counts[cut]; }
                job.cut = cut;
                job.cutLimit = limit;
                job.cutEmitted = 0;
                job.total = acc;
                counts[cut] = limit;

                // Emit far -> near: farthest kept bucket gets slot 0.
                let running = 0;
                for (let b = cut; b >= 0; b--) {
                    const c = counts[b];
                    counts[b] = running;
                    running += c;
                }
                job.cursor = 0;
                job.phase = job.total > 0 ? 'scatter' : 'upload';

            } else if (job.phase === 'scatter') {
                const order = this.order, cut = job.cut, lim = job.cutLimit;
                let i = job.cursor;
                const end = Math.min(n, i + SORT_CHUNK);
                let emitted = job.cutEmitted;
                for (; i < end; i++) {
                    const b = bucketOf[i];
                    if (b > cut) continue;          // culled (0xFFFF) or beyond budget
                    if (b === cut) {
                        if (emitted >= lim) continue;
                        emitted++;
                    }
                    order[counts[b]++] = i;
                }
                job.cutEmitted = emitted;
                job.cursor = i;
                if (i >= n) job.phase = 'upload';

            } else if (job.phase === 'upload') {
                const gl = this.gl;
                const total = Math.min(job.total, this.maxOrder);
                if (total > 0) {
                    gl.bindBuffer(gl.ARRAY_BUFFER, this.orderVBO);
                    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.order, 0, total);
                    gl.bindBuffer(gl.ARRAY_BUFFER, null);
                }
                this.visibleCount = total;
                this.orderValid = true;
                this.lastSortFwd = job.fwd.slice();
                this.lastSortEye = job.eye.slice();
                this.lastSortAt = performance.now();
                this.sortDirty = false;
                job.phase = 'idle';
                return true;
            }

            if (performance.now() - t0 >= budgetMs) return false;
        }
        return true;
    }

    /* -------------------------------------------------------- interaction */

    markMoving() { this.movingUntil = performance.now() + 120; }

    initEvents() {
        window.addEventListener('resize', () => this.resize());
        window.addEventListener('orientationchange', () => setTimeout(() => this.resize(), 120));

        this.canvas.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            this.contextLost = true;
            UI.fatal('The graphics context was lost (the device reclaimed GPU memory).\n\nReopen the model to try again.');
        }, false);

        this.initPointer();

        $('btn-reset').addEventListener('click', () => this.resetCamera());
        $('btn-toggle-bg').addEventListener('click', () => this.applyTheme(!this.themeDark));
        $('btn-measure').addEventListener('click', () => this.toggleMeasure());
        $('btn-crop').addEventListener('click', () => this.cropFloor());
        $('btn-clean').addEventListener('click', () => this.cleanFloaters());
        $('btn-save').addEventListener('click', () => this.saveModel());
        $('btn-restore').addEventListener('click', () => this.restoreOriginal());
        $('btn-ar').addEventListener('click', () => {
            if (window.AndroidBridge && window.AndroidBridge.launchAR) window.AndroidBridge.launchAR();
            else UI.toast('AR placement requires an ARCore device');
        });
    }

    /**
     * Unified pointer handling: 1 pointer orbits, 2 pointers pinch-zoom and pan.
     * A short, small-movement press is treated as a tap (used by Measure).
     */
    initPointer() {
        const cv = this.canvas;
        const pts = new Map();
        let mode = 'none';
        let lastX = 0, lastY = 0, lastDist = 0, lastMidX = 0, lastMidY = 0;
        let downT = 0, downX = 0, downY = 0, travel = 0;

        const mid = () => {
            let sx = 0, sy = 0, n = 0;
            pts.forEach(p => { sx += p.x; sy += p.y; n++; });
            return [sx / n, sy / n, n];
        };
        const dist = () => {
            const a = [...pts.values()];
            return Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y);
        };

        const down = (e) => {
            try { cv.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
            pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
            this.vel.theta = this.vel.phi = this.vel.radius = 0;
            this.vel.tx = this.vel.ty = this.vel.tz = 0;
            this.pointerActive = true;
            if (pts.size === 1) {
                mode = 'orbit';
                lastX = e.clientX; lastY = e.clientY;
                downT = performance.now(); downX = e.clientX; downY = e.clientY; travel = 0;
            } else if (pts.size === 2) {
                mode = 'pinch';
                lastDist = dist();
                const m = mid(); lastMidX = m[0]; lastMidY = m[1];
            }
            this.markMoving();
        };

        const move = (e) => {
            if (!pts.has(e.pointerId)) return;
            pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
            e.preventDefault();

            if (mode === 'orbit' && pts.size === 1) {
                const dx = e.clientX - lastX, dy = e.clientY - lastY;
                lastX = e.clientX; lastY = e.clientY;
                travel += Math.abs(dx) + Math.abs(dy);
                const k = 2.6 / Math.max(320, this.cssH);
                this.vel.theta = -dx * k;
                this.vel.phi = -dy * k;
                this.applyOrbit(-dx * k, -dy * k);
            } else if (mode === 'pinch' && pts.size >= 2) {
                const d = dist();
                if (lastDist > 0 && d > 0) {
                    const f = lastDist / d;
                    this.applyZoom(f);
                    this.vel.radius = this.camera.radius * (f - 1) * 0.6;
                }
                lastDist = d;
                const m = mid();
                this.applyPan(m[0] - lastMidX, m[1] - lastMidY);
                lastMidX = m[0]; lastMidY = m[1];
            }
            this.markMoving();
        };

        const up = (e) => {
            const had = pts.has(e.pointerId);
            pts.delete(e.pointerId);
            try { if (cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }

            if (had && mode === 'orbit' && pts.size === 0) {
                const dt = performance.now() - downT;
                const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
                if (dt < 400 && moved < 12 && travel < 24) {
                    this.vel.theta = this.vel.phi = 0;
                    this.onTap(e.clientX, e.clientY);
                }
            }
            if (pts.size === 0) { mode = 'none'; this.pointerActive = false; }
            else if (pts.size === 1) {
                mode = 'orbit';
                const p = [...pts.values()][0];
                lastX = p.x; lastY = p.y; travel = 1e9;
            }
            this.markMoving();
        };

        cv.addEventListener('pointerdown', down, { passive: false });
        cv.addEventListener('pointermove', move, { passive: false });
        window.addEventListener('pointerup', up, { passive: false });
        window.addEventListener('pointercancel', up, { passive: false });

        window.addEventListener('wheel', (e) => {
            this.applyZoom(Math.exp(e.deltaY * 0.0013));
            this.markMoving();
        }, { passive: true });
    }

    applyOrbit(dTheta, dPhi) {
        const c = this.camera;
        c.theta += dTheta;
        c.phi = clamp(c.phi + dPhi, 0.04, Math.PI - 0.04);
        this.sortDirty = true;
    }

    applyZoom(factor) {
        const c = this.camera;
        const lo = Math.max(this.scene.radius * 0.03, c.near * 3);
        const hi = this.scene.maxRadius * 30 + 1;
        c.radius = clamp(c.radius * factor, lo, hi);
        c.far = Math.max(c.far, c.radius * 4);
        this.sortDirty = true;
    }

    /** Screen-space pan: dragging keeps the grabbed point roughly under the finger. */
    applyPan(dxPx, dyPx) {
        if (!dxPx && !dyPx) return;
        const b = this.lastBasis || this.cameraBasis();
        const c = this.camera;
        const worldPerPx = 2 * c.radius * Math.tan(c.fov / 2) / Math.max(1, this.cssH);
        const kx = -dxPx * worldPerPx, ky = dyPx * worldPerPx;
        for (let i = 0; i < 3; i++) c.target[i] += b.right[i] * kx + b.up[i] * ky;
        this.vel.tx = b.right[0] * kx + b.up[0] * ky;
        this.vel.ty = b.right[1] * kx + b.up[1] * ky;
        this.vel.tz = b.right[2] * kx + b.up[2] * ky;
        this.sortDirty = true;
    }

    /** Inertial glide after the finger leaves the screen. */
    applyInertia(dt, touching) {
        if (touching) return false;
        const v = this.vel;
        const damp = Math.pow(0.90, dt / 16.6667);
        let active = false;
        if (Math.abs(v.theta) > 1e-5 || Math.abs(v.phi) > 1e-5) {
            this.applyOrbit(v.theta, v.phi);
            v.theta *= damp; v.phi *= damp;
            active = true;
        } else { v.theta = v.phi = 0; }

        if (Math.abs(v.radius) > this.scene.radius * 1e-4) {
            this.camera.radius = clamp(this.camera.radius + v.radius,
                Math.max(this.scene.radius * 0.03, this.camera.near * 3),
                this.scene.maxRadius * 30 + 1);
            v.radius *= damp;
            active = true;
            this.sortDirty = true;
        } else { v.radius = 0; }

        const tmag = Math.abs(v.tx) + Math.abs(v.ty) + Math.abs(v.tz);
        if (tmag > this.scene.radius * 1e-4) {
            this.camera.target[0] += v.tx; this.camera.target[1] += v.ty; this.camera.target[2] += v.tz;
            v.tx *= damp; v.ty *= damp; v.tz *= damp;
            active = true;
            this.sortDirty = true;
        } else { v.tx = v.ty = v.tz = 0; }

        return active;
    }

    resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, this.caps.lowRam ? 1.5 : 2.25);
        this.cssW = Math.max(1, window.innerWidth);
        this.cssH = Math.max(1, window.innerHeight);
        const w = Math.round(this.cssW * dpr), h = Math.round(this.cssH * dpr);
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
        const svg = $('measure-svg');
        svg.setAttribute('viewBox', '0 0 ' + this.cssW + ' ' + this.cssH);
        svg.setAttribute('width', this.cssW);
        svg.setAttribute('height', this.cssH);
        this.sortDirty = true;
    }

    /* ------------------------------------------------------------ render */

    renderLoop(now) {
        this.rafId = requestAnimationFrame((t) => this.renderLoop(t));
        if (this.contextLost) return;

        const gl = this.gl;
        let dt = now - this.lastFrame;
        this.lastFrame = now;
        if (!(dt > 0) || dt > 500) dt = 16.7;
        this.frameMs = this.frameMs * 0.88 + dt * 0.12;
        this.fps = 1000 / Math.max(1, this.frameMs);

        const touching = !!this.pointerActive;
        const gliding = this.applyInertia(dt, touching);
        const moving = touching || gliding || now < this.movingUntil;

        const bg = this.themeDark ? 0.043 : 0.925;
        gl.clearColor(bg, bg, bg, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        if (this.splatCount > 0) {
            const b = this.cameraBasis();
            this.lastBasis = b;
            const aspect = this.canvas.width / Math.max(1, this.canvas.height);
            const view = this.viewMatrix(b);
            const proj = this.projMatrix(aspect);
            this.p00 = proj[0];
            this.p11 = proj[5];

            // Cull + sort, incrementally. More time when settled, very little
            // while the camera is in motion so frames never hitch.
            this.maybeStartSortJob(now, moving);
            if (this.job.phase !== 'idle') {
                this.pumpSortJob(moving ? 2.5 : (this.frameMs < 20 ? 7 : 4));
            }

            this.tuneBudget(now, dt, moving);

            const drawCount = Math.min(this.visibleCount, Math.max(1, Math.round(this.budget)));
            if (this.orderValid && drawCount > 0) {
                gl.useProgram(this.program);
                gl.uniformMatrix4fv(this.uni.view, false, view);
                gl.uniformMatrix4fv(this.uni.proj, false, proj);
                gl.uniform2f(this.uni.viewport, this.canvas.width, this.canvas.height);
                gl.uniform2f(this.uni.focal,
                    proj[0] * this.canvas.width * 0.5,
                    proj[5] * this.canvas.height * 0.5);
                gl.uniform1i(this.uni.texWidth, this.texWidth);

                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, this.splatTex);

                gl.bindVertexArray(this.vao);
                // The list runs far -> near, so "the nearest N" is a suffix: a
                // byte offset into the index buffer. No re-upload, no CPU work.
                const firstByte = (this.visibleCount - drawCount) * 4;
                gl.bindBuffer(gl.ARRAY_BUFFER, this.orderVBO);
                gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, firstByte);
                gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, drawCount);
                gl.bindVertexArray(null);
            }
            this.drawnCount = drawCount;

            if (this.measurePoints.length) this.updateMeasureOverlay();
        }

        if (now - this.lastHudAt > 260) {
            this.lastHudAt = now;
            this.updateHud();
        }
    }

    /**
     * Dynamic LOD. `autoBudget` tracks what this GPU can sustain while the
     * camera is SETTLED, targeting ~35 fps rather than 60.
     *
     * Inspecting a reconstruction is not a game: holding 60 fps by drawing 19%
     * of the Gaussians makes a sharp model look like fog, which is exactly the
     * blur users report. When the camera is still, detail is worth far more than
     * frame rate, so the budget is allowed to grow until frames cost ~28 ms.
     * Motion still drops to a reduced budget, where the blur is hidden anyway.
     * `budget` eases toward a reduced target while the camera moves and back to
     * full density once it settles.
     */
    tuneBudget(now, dt, moving) {
        const n = this.visibleCount || this.splatCount;

        // Only measure while the camera is settled: that is the only time the
        // frame time reflects full density rather than the reduced motion LOD.
        if (moving || this.job.phase !== 'idle') {
            this.tuneAccum = 0; this.tuneFrames = 0; this.lastTune = now;
        } else {
            this.tuneAccum += dt;
            this.tuneFrames++;
            if (now - this.lastTune > 550 && this.tuneFrames > 10) {
                const avg = this.tuneAccum / this.tuneFrames;
                this.tuneAccum = 0; this.tuneFrames = 0; this.lastTune = now;
                // Frames pinned to the refresh interval mean spare GPU time, so
                // the budget can grow back; anything slower means back off.
                if (avg > 40) this.autoBudget *= 0.85;          // below ~25 fps
                else if (avg > 28) this.autoBudget *= 0.95;        // below ~36 fps
                else if (avg < 24 && this.budget > this.autoBudget * 0.9) this.autoBudget *= 1.15;
                this.autoBudget = clamp(this.autoBudget, 40000, this.caps.maxSplats);
            }
        }

        const full = Math.min(n, this.autoBudget);
        const target = moving ? Math.max(40000, full * 0.42) : full;

        // Drop fast (protect the frame rate), restore gently (no visible pop).
        const k = target < this.budget ? 0.35 : 0.055;
        this.budget += (target - this.budget) * (1 - Math.pow(1 - k, Math.max(0.5, this.frameMs / 16.7)));
        this.budget = clamp(this.budget, 1, n || 1);
        this.lodActive = this.budget < n * 0.985;
    }

    updateHud() {
        if (!this.splatCount) return;
        $('stat-fps').textContent = this.fps > 0 ? String(Math.round(this.fps)) : '--';
        $('stat-shown').textContent = fmt(this.drawnCount || 0);
        $('stat-total').textContent = fmt(this.splatCount);
        const chip = $('chip-stats');
        chip.classList.toggle('hide-lod', !this.lodActive);
        if (this.lodActive) {
            $('stat-lod').textContent = 'LOD ' +
                Math.round(100 * (this.drawnCount || 0) / Math.max(1, this.splatCount)) + '%';
        }
    }

    /* ------------------------------------------------------------- theme */

    applyTheme(dark) {
        this.themeDark = !!dark;
        document.body.classList.toggle('theme-dark', this.themeDark);
        $('theme-ic').innerHTML = this.themeDark ? '&#9788;' : '&#9790;';
        $('theme-label').textContent = this.themeDark ? 'Light' : 'Dark';
        try {
            if (window.AndroidBridge && window.AndroidBridge.onThemeChanged) {
                window.AndroidBridge.onThemeChanged(this.themeDark);
            }
        } catch (e) { /* non-fatal */ }
    }

    /* --------------------------------------------------------- measuring */

    toggleMeasure() {
        this.measuring = !this.measuring;
        this.measurePoints = [];
        $('btn-measure').classList.toggle('on', this.measuring);
        this.updateMeasureOverlay();
        if (this.measuring) {
            UI.hint('Tap two points on the model');
        } else {
            UI.hint(null);
        }
    }

    onTap(clientX, clientY) {
        if (!this.measuring || !this.splatCount) return;
        const rect = this.canvas.getBoundingClientRect();
        const hit = this.raycast(clientX - rect.left, clientY - rect.top);
        if (!hit) {
            UI.toast('No surface there - tap directly on the model');
            return;
        }
        // A third tap starts a fresh measurement.
        if (this.measurePoints.length >= 2) this.measurePoints = [];
        this.measurePoints.push(hit);

        if (this.measurePoints.length === 1) {
            this.measureDistance = 0;
            UI.hint('Tap the second point');
        } else {
            const p = this.measurePoints[0], q = this.measurePoints[1];
            const d = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
            this.measureDistance = d;
            UI.hint('Distance: ' + metersLabel(d));
            try {
                if (window.AndroidBridge && window.AndroidBridge.showMeasurementToast) {
                    window.AndroidBridge.showMeasurementToast(d);
                }
            } catch (e) { /* non-fatal */ }
        }
        this.updateMeasureOverlay();
    }

    /**
     * Real screen-space raycast into the splat cloud.
     *
     * Stage 1 rejects almost everything with a cheap ray/bounding-sphere test.
     * Stage 2 evaluates the true anisotropic Gaussian along the ray for the few
     * survivors (closest approach in the splat's own Mahalanobis metric has a
     * closed form), then walks them front-to-back accumulating transmittance and
     * returns the depth at which the surface becomes opaque. That is the same
     * "median depth" a 3DGS depth render produces, so the returned point is a
     * real point on the reconstructed surface - and because ARCore anchors the
     * capture in metres, the distance between two such points is metric.
     */
    raycast(cssX, cssY) {
        const b = this.lastBasis;
        if (!b || !this.f32) return null;

        const ndcX = (cssX / this.cssW) * 2 - 1;
        const ndcY = 1 - (cssY / this.cssH) * 2;
        const cx = ndcX / this.p00, cy = ndcY / this.p11;
        let dx = b.right[0] * cx + b.up[0] * cy + b.fwd[0];
        let dy = b.right[1] * cx + b.up[1] * cy + b.fwd[1];
        let dz = b.right[2] * cx + b.up[2] * cy + b.fwd[2];
        const dl = Math.hypot(dx, dy, dz) || 1;
        dx /= dl; dy /= dl; dz /= dl;

        const ox = b.eye[0], oy = b.eye[1], oz = b.eye[2];
        const f = this.f32, u8 = this.u8, n = this.splatCount;
        const near = this.camera.near;

        // ---- stage 1: broad phase ----
        const MAXC = 24576;
        const candT = new Float64Array(MAXC);
        const candI = new Int32Array(MAXC);
        let nc = 0;
        for (let i = 0; i < n; i++) {
            const o = i * WORDS_PER_SPLAT;
            const wx = f[o] - ox, wy = f[o + 1] - oy, wz = f[o + 2] - oz;
            const t = wx * dx + wy * dy + wz * dz;
            if (t <= near) continue;
            let sx = f[o + 3]; if (sx < 0) sx = -sx;
            let sy = f[o + 4]; if (sy < 0) sy = -sy;
            let sz = f[o + 5]; if (sz < 0) sz = -sz;
            let r = sx > sy ? sx : sy; if (sz > r) r = sz;
            r *= 2.5;
            const perp2 = (wx * wx + wy * wy + wz * wz) - t * t;
            if (perp2 > r * r) continue;
            if (u8[i * SPLAT_BYTES + 27] < 10) continue;   // effectively invisible
            if (nc < MAXC) { candT[nc] = t; candI[nc] = i; nc++; }
        }
        if (nc === 0) return null;

        // ---- stage 2: narrow phase, exact anisotropic Gaussian ----
        const ta = candT;
        const idx = new Int32Array(nc);
        for (let k = 0; k < nc; k++) idx[k] = k;
        idx.sort((a, c) => ta[a] - ta[c]);   // front to back

        let transmittance = 1.0;
        let acc = 0, accT = 0;
        for (let k = 0; k < nc; k++) {
            const ci = idx[k];
            const i = candI[ci];
            const o = i * WORDS_PER_SPLAT;
            const q = i * SPLAT_BYTES;

            // quaternion, scalar first, q = r*128 + 128
            let qw = u8[q + 28] / 128 - 1, qx = u8[q + 29] / 128 - 1,
                qy = u8[q + 30] / 128 - 1, qz = u8[q + 31] / 128 - 1;
            const ql = Math.hypot(qw, qx, qy, qz) || 1;
            qw /= ql; qx /= ql; qy /= ql; qz /= ql;

            // Rotation matrix columns (world axes of the ellipsoid).
            const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
            const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
            const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);

            const s0 = Math.max(1e-6, Math.abs(f[o + 3]));
            const s1 = Math.max(1e-6, Math.abs(f[o + 4]));
            const s2 = Math.max(1e-6, Math.abs(f[o + 5]));

            const wx = ox - f[o], wy = oy - f[o + 1], wz = oz - f[o + 2];
            // a = S^-1 R^T (origin - centre), b = S^-1 R^T dir
            const a0 = (r00 * wx + r10 * wy + r20 * wz) / s0;
            const a1 = (r01 * wx + r11 * wy + r21 * wz) / s1;
            const a2 = (r02 * wx + r12 * wy + r22 * wz) / s2;
            const b0 = (r00 * dx + r10 * dy + r20 * dz) / s0;
            const b1 = (r01 * dx + r11 * dy + r21 * dz) / s1;
            const b2 = (r02 * dx + r12 * dy + r22 * dz) / s2;

            const bb = b0 * b0 + b1 * b1 + b2 * b2;
            if (bb < 1e-12) continue;
            const ab = a0 * b0 + a1 * b1 + a2 * b2;
            const t = -ab / bb;                       // closest approach, in metres
            if (t <= near) continue;
            const aa = a0 * a0 + a1 * a1 + a2 * a2;
            const m2 = Math.max(0, aa - ab * ab / bb);   // squared Mahalanobis distance
            if (m2 > 9) continue;                        // beyond 3 sigma

            const alpha = Math.min(0.99, (u8[q + 27] / 255) * Math.exp(-0.5 * m2));
            if (alpha < 0.02) continue;

            const w = transmittance * alpha;
            acc += w;
            accT += w * t;
            transmittance *= (1 - alpha);
            if (transmittance < 0.35) break;             // surface reached
        }

        if (acc < 0.08) return null;                     // only wisps along this ray
        const t = accT / acc;
        return [ox + dx * t, oy + dy * t, oz + dz * t];
    }

    updateMeasureOverlay() {
        const svg = $('measure-svg');
        const label = $('measure-label');
        const pts = this.measurePoints;
        if (!this.measuring || pts.length === 0) {
            svg.classList.remove('show');
            label.classList.remove('show');
            return;
        }
        svg.classList.add('show');

        const a = this.projectPoint(pts[0]);
        const bpt = pts.length > 1 ? this.projectPoint(pts[1]) : null;

        const set = (id, p) => {
            const o = $(id + '-outer'), n = $(id + '-inner');
            if (!p) { o.setAttribute('cx', -99); n.setAttribute('cx', -99); return; }
            o.setAttribute('cx', p[0]); o.setAttribute('cy', p[1]);
            n.setAttribute('cx', p[0]); n.setAttribute('cy', p[1]);
        };
        set('m0', a);
        set('m1', bpt);

        const line = $('measure-line'), halo = $('measure-line-halo');
        if (a && bpt) {
            line.setAttribute('x1', a[0]); line.setAttribute('y1', a[1]);
            line.setAttribute('x2', bpt[0]); line.setAttribute('y2', bpt[1]);
            halo.setAttribute('x1', a[0]); halo.setAttribute('y1', a[1]);
            halo.setAttribute('x2', bpt[0]); halo.setAttribute('y2', bpt[1]);
            const txt = metersLabel(this.measureDistance || 0);
            if (label.textContent !== txt) label.textContent = txt;
            label.classList.add('show');
            // transform only -- never reads layout, so no thrash in the raf loop
            label.style.transform = 'translate(' + ((a[0] + bpt[0]) / 2) + 'px,' +
                ((a[1] + bpt[1]) / 2 - 34) + 'px) translateX(-50%)';
        } else {
            line.setAttribute('x2', line.getAttribute('x1'));
            halo.setAttribute('x2', halo.getAttribute('x1'));
            label.classList.remove('show');
        }
    }

    /* ------------------------------------------------------------- tools */

    /**
     * Build a new 32-byte buffer from the splats `keep(i)` accepts. Two passes,
     * word-wise copy - no per-splat subarray allocation.
     */
    filterSplats(keep) {
        const n = this.splatCount;
        const src = new Uint32Array(this.splatBuffer, 0, n * WORDS_PER_SPLAT);
        let kept = 0;
        const mask = new Uint8Array(n);
        for (let i = 0; i < n; i++) { if (keep(i)) { mask[i] = 1; kept++; } }
        if (kept === 0) return null;
        if (kept === n) return this.splatBuffer;

        const out = new Uint32Array(kept * WORDS_PER_SPLAT);
        let w = 0;
        for (let i = 0; i < n; i++) {
            if (!mask[i]) continue;
            const s = i * WORDS_PER_SPLAT;
            out[w] = src[s]; out[w + 1] = src[s + 1]; out[w + 2] = src[s + 2]; out[w + 3] = src[s + 3];
            out[w + 4] = src[s + 4]; out[w + 5] = src[s + 5]; out[w + 6] = src[s + 6]; out[w + 7] = src[s + 7];
            w += WORDS_PER_SPLAT;
        }
        return out.buffer;
    }

    applyEdit(buffer, label) {
        if (!buffer) { UI.toast('That would remove every splat - nothing changed'); return; }
        const before = this.splatCount;
        this.edited = true;
        this.keepCameraOnLoad = true;      // an edit must not re-frame the view
        try {
            this.setSplatData(buffer);
        } catch (e) {
            console.error(e);
            UI.toast('Edit failed: ' + (e && e.message ? e.message : e));
            return;
        }
        UI.toast(label + ' - ' + fmt(before - this.splatCount) + ' removed, ' + fmt(this.splatCount) + ' left', 3000);
    }

    /** Remove the ground plane: the lowest slice of a robust height histogram. */
    cropFloor() {
        if (!this.splatCount) return;
        const f = this.f32, n = this.splatCount;

        // Robust low/high percentiles of Y so a single stray splat cannot set
        // the threshold (the old min/max version was trivially skewed).
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < n; i++) { const y = f[i * WORDS_PER_SPLAT + 1]; if (y < lo) lo = y; if (y > hi) hi = y; }
        if (!(hi > lo)) { UI.toast('Model is flat - nothing to crop'); return; }
        const HB = 1024, hist = new Uint32Array(HB), sc = (HB - 1) / (hi - lo);
        for (let i = 0; i < n; i++) hist[((f[i * WORDS_PER_SPLAT + 1] - lo) * sc) | 0]++;
        let acc = 0, p01 = 0, p99 = HB - 1;
        for (let b = 0; b < HB; b++) { acc += hist[b]; if (acc >= n * 0.01) { p01 = b; break; } }
        acc = 0;
        for (let b = HB - 1; b >= 0; b--) { acc += hist[b]; if (acc >= n * 0.01) { p99 = b; break; } }
        const yLo = lo + p01 / sc, yHi = lo + p99 / sc;
        const threshold = yLo + (yHi - yLo) * 0.10;

        this.applyEdit(this.filterSplats((i) => f[i * WORDS_PER_SPLAT + 1] >= threshold), 'Floor cropped');
    }

    /** Remove near-transparent wisps, blown-up splats and distant floaters. */
    cleanFloaters() {
        if (!this.splatCount) return;
        const f = this.f32, u8 = this.u8, n = this.splatCount;
        // Scene-relative thresholds: an absolute "2 m" limit is meaningless for a
        // model of arbitrary scale.
        const maxScale = Math.max(this.scene.medScale * 12, this.scene.radius * 0.08);
        const c = this.scene.center;
        const maxR2 = Math.pow(this.scene.radius * 1.8, 2);

        this.applyEdit(this.filterSplats((i) => {
            if (u8[i * SPLAT_BYTES + 27] < 15) return false;
            const o = i * WORDS_PER_SPLAT;
            const sx = Math.abs(f[o + 3]), sy = Math.abs(f[o + 4]), sz = Math.abs(f[o + 5]);
            if (!(sx > 0) && !(sy > 0) && !(sz > 0)) return false;
            if (sx > maxScale || sy > maxScale || sz > maxScale) return false;
            const dx = f[o] - c[0], dy = f[o + 1] - c[1], dz = f[o + 2] - c[2];
            if (dx * dx + dy * dy + dz * dz > maxR2) return false;
            return true;
        }), 'Floaters pruned');
    }

    restoreOriginal() {
        if (!this.sourceUrl) return;
        this.edited = false;
        this.measurePoints = [];
        this.loadSplat(this.sourceUrl, true);
    }

    /**
     * Write the current (possibly edited) buffer back to disk as a valid .splat.
     * Streamed in small chunks so a 60 MB model never needs a 80 MB base64
     * string in the WebView heap.
     */
    saveModel() {
        if (!this.splatBuffer || this.saving) return;
        const B = window.AndroidBridge;
        if (!B || !B.beginSave) { UI.toast('Saving is only available in the app'); return; }

        const bytes = new Uint8Array(this.splatBuffer, 0, this.splatCount * SPLAT_BYTES);
        const total = bytes.byteLength;
        if (total === 0 || total % SPLAT_BYTES !== 0) { UI.toast('Nothing valid to save'); return; }

        let started;
        try {
            started = B.beginSave(this.splatCount);
        } catch (e) { UI.toast('Save failed to start'); return; }
        if (!started) { UI.toast('Save failed to start'); return; }

        this.saving = true;
        $('btn-save').setAttribute('disabled', 'true');
        UI.setLoading('Saving model&hellip;', '');
        UI.setProgress(0);

        const CHUNK = 393216;               // 384 KB, a multiple of 3 and of 32
        let off = 0;
        const step = () => {
            try {
                const end = Math.min(total, off + CHUNK);
                let s = '';
                const sub = bytes.subarray(off, end);
                // 8 KB at a time keeps the apply() argument list small.
                for (let i = 0; i < sub.length; i += 8192) {
                    s += String.fromCharCode.apply(null, sub.subarray(i, Math.min(i + 8192, sub.length)));
                }
                B.appendSaveChunk(btoa(s));
                off = end;
                UI.setProgress(off / total);
                $('loading-sub').textContent = (off / 1048576).toFixed(1) + ' / ' + (total / 1048576).toFixed(1) + ' MB';
                if (off < total) {
                    setTimeout(step, 0);
                } else {
                    const name = B.endSave();
                    this.saving = false;
                    $('btn-save').removeAttribute('disabled');
                    UI.hideLoading();
                    UI.toast(name ? ('Saved ' + name + ' (' + fmt(this.splatCount) + ' splats)')
                                  : 'Saved ' + fmt(this.splatCount) + ' splats', 3600);
                }
            } catch (e) {
                console.error(e);
                try { B.abortSave(); } catch (_) { /* ignore */ }
                this.saving = false;
                $('btn-save').removeAttribute('disabled');
                UI.hideLoading();
                UI.toast('Save failed: ' + (e && e.message ? e.message : e), 3600);
            }
        };
        setTimeout(step, 0);
    }
}

/* --------------------------------------------------------------- bootstrap */

function boot() {
    let v;
    try {
        v = new GaussianSplatViewer();
    } catch (e) {
        console.error(e);
        UI.fatal('The viewer failed to start: ' + (e && e.message ? e.message : e));
        return;
    }
    window.viewer = v;
    if (!v.ok) return;

    let loaded = false;
    try {
        if (window.AndroidBridge && window.AndroidBridge.hasCustomModel && window.AndroidBridge.hasCustomModel()) {
            const name = window.AndroidBridge.getModelName();
            if (name) $('model-name').textContent = name;
            // Streamed from the app over the same origin - no base64 bridge.
            v.loadSplat(window.AndroidBridge.getModelUrl());
            loaded = true;
        }
    } catch (e) {
        console.error('AndroidBridge load error: ' + e);
        UI.setLoading('Could not reach the model.', String(e));
    }

    if (!loaded) {
        const url = new URLSearchParams(window.location.search).get('url');
        if (url) v.loadSplat(url);
        else UI.setLoading('No 3D model loaded.', 'Open a capture from the library to view it.');
    }
}

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', boot);
} else {
    boot();
}

window.loadSplatFromUrl = function (url, name) {
    if (window.viewer && window.viewer.ok) {
        $('model-name').textContent = name || '3D Gaussian Splat';
        window.viewer.edited = false;
        window.viewer.loadSplat(url);
    }
};

})();
