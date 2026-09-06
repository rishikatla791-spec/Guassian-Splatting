// Mobile 3D Gaussian Splatting WebGL2 Engine
class GaussianSplatViewer {
    constructor() {
        this.canvas = document.getElementById('gl-canvas');
        this.gl = this.canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance' });
        
        if (!this.gl) {
            document.getElementById('loading-text').innerText = 'WebGL2 not supported on this device.';
            return;
        }

        this.splatCount = 0;
        this.camera = {
            target: [0, 0, 0],
            radius: 3.5,
            theta: 0.0,
            phi: Math.PI / 4,
            fov: 50 * Math.PI / 180
        };

        this.touch = {
            active: false,
            mode: 'none',
            lastX: 0,
            lastY: 0,
            lastDist: 0
        };

        this.bgDark = true;
        this.initShaders();
        this.initBuffers();
        this.initEvents();
        this.resize();

        const urlParams = new URLSearchParams(window.location.search);
        const modelUrl = urlParams.get('url') || 'people_me.splat';
        this.loadSplat(modelUrl);

        this.lastTime = performance.now();
        this.frameCount = 0;
        this.fps = 60;
        
        requestAnimationFrame((t) => this.renderLoop(t));
    }

    initShaders() {
        const gl = this.gl;
        const vsSource = `#version 300 es
        precision highp float;
        
        layout(location = 0) in vec2 a_quad;
        layout(location = 1) in vec3 a_center;
        layout(location = 2) in vec3 a_scale;
        layout(location = 3) in vec4 a_color;
        layout(location = 4) in vec4 a_rotation;

        uniform mat4 u_view;
        uniform mat4 u_proj;
        uniform vec2 u_viewport;

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
            v_color = a_color;
            v_coord = a_quad;

            vec4 viewPos = u_view * vec4(a_center, 1.0);
            
            // Unpack rotation [0, 255] -> [-1, 1]
            vec4 rot = a_rotation * (2.0 / 255.0) - 1.0;
            mat3 R = quatToMat(rot);
            mat3 S = mat3(a_scale.x, 0.0, 0.0, 0.0, a_scale.y, 0.0, 0.0, 0.0, a_scale.z);
            mat3 M = mat3(u_view) * R * S;

            vec2 offset = a_quad * (length(a_scale) * 2.0 + 0.03);
            vec4 pos = viewPos + vec4(offset.x, offset.y, 0.0, 0.0);
            
            gl_Position = u_proj * pos;
        }`;

        const fsSource = `#version 300 es
        precision highp float;

        in vec4 v_color;
        in vec2 v_coord;
        out vec4 fragColor;

        void main() {
            float distSq = dot(v_coord, v_coord);
            if (distSq > 1.0) discard;
            
            float alpha = exp(-2.0 * distSq) * v_color.a;
            if (alpha < 0.02) discard;
            
            fragColor = vec4(v_color.rgb * alpha, alpha);
        }`;

        const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);

        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);

        this.u_view = gl.getUniformLocation(this.program, 'u_view');
        this.u_proj = gl.getUniformLocation(this.program, 'u_proj');
        this.u_viewport = gl.getUniformLocation(this.program, 'u_viewport');
    }

    compileShader(type, src) {
        const gl = this.gl;
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(s));
        }
        return s;
    }

    initBuffers() {
        const gl = this.gl;
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        const quadVerts = new Float32Array([ -1, -1,  1, -1, -1,  1,  1,  1 ]);
        this.quadVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        this.instanceVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVBO);

        // Attribute 1: center (vec3, 12B)
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 0);
        gl.vertexAttribDivisor(1, 1);

        // Attribute 2: scale (vec3, 12B)
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 32, 12);
        gl.vertexAttribDivisor(2, 1);

        // Attribute 3: color (RGBA, 4B)
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, 32, 24);
        gl.vertexAttribDivisor(3, 1);

        // Attribute 4: rotation (quat, 4B)
        gl.enableVertexAttribArray(4);
        gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, false, 32, 28);
        gl.vertexAttribDivisor(4, 1);

        gl.bindVertexArray(null);
    }

    async loadSplat(url) {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('loading-text').innerText = 'Loading 3D Splatting Scene...';
        
        try {
            const resp = await fetch(url);
            if (!resp.ok) throw new Error('Failed to fetch ' + url);
            const buffer = await resp.arrayBuffer();
            this.setSplatData(buffer);
        } catch (e) {
            console.log('Model load fallback: ' + e);
            document.getElementById('loading-text').innerText = 'Ready. Select a model to view.';
            setTimeout(() => { document.getElementById('loading').style.display = 'none'; }, 1000);
        }
    }

    setSplatData(arrayBuffer) {
        const gl = this.gl;
        this.splatCount = Math.floor(arrayBuffer.byteLength / 32);
        
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceVBO);
        gl.bufferData(gl.ARRAY_BUFFER, arrayBuffer, gl.STATIC_DRAW);

        document.getElementById('loading').style.display = 'none';
        document.getElementById('model-stats').innerText = `${this.splatCount.toLocaleString()} Gaussians | 60 FPS`;
    }

    initEvents() {
        window.addEventListener('resize', () => this.resize());

        this.canvas.addEventListener('touchstart', (e) => this.onTouchStart(e), { passive: false });
        this.canvas.addEventListener('touchmove', (e) => this.onTouchMove(e), { passive: false });
        this.canvas.addEventListener('touchend', (e) => this.onTouchEnd(e), { passive: false });

        document.getElementById('btn-reset').addEventListener('click', () => {
            this.camera.radius = 3.5;
            this.camera.theta = 0.0;
            this.camera.phi = Math.PI / 4;
            this.camera.target = [0, 0, 0];
        });

        document.getElementById('btn-toggle-bg').addEventListener('click', () => {
            this.bgDark = !this.bgDark;
        });
    }

    onTouchStart(e) {
        e.preventDefault();
        if (e.touches.length === 1) {
            this.touch.mode = 'rotate';
            this.touch.lastX = e.touches[0].clientX;
            this.touch.lastY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
            this.touch.mode = 'pinch';
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            this.touch.lastDist = Math.sqrt(dx * dx + dy * dy);
        }
    }

    onTouchMove(e) {
        e.preventDefault();
        if (this.touch.mode === 'rotate' && e.touches.length === 1) {
            const dx = e.touches[0].clientX - this.touch.lastX;
            const dy = e.touches[0].clientY - this.touch.lastY;
            this.touch.lastX = e.touches[0].clientX;
            this.touch.lastY = e.touches[0].clientY;

            this.camera.theta -= dx * 0.006;
            this.camera.phi = Math.max(0.05, Math.min(Math.PI - 0.05, this.camera.phi - dy * 0.006));
        } else if (this.touch.mode === 'pinch' && e.touches.length === 2) {
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const delta = dist - this.touch.lastDist;
            this.touch.lastDist = dist;

            this.camera.radius = Math.max(0.5, Math.min(15.0, this.camera.radius - delta * 0.015));
        }
    }

    onTouchEnd(e) {
        if (e.touches.length === 0) {
            this.touch.mode = 'none';
        }
    }

    resize() {
        this.canvas.width = window.innerWidth * window.devicePixelRatio;
        this.canvas.height = window.innerHeight * window.devicePixelRatio;
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    renderLoop(now) {
        const gl = this.gl;
        this.frameCount++;
        if (now - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = now;
            if (this.splatCount > 0) {
                document.getElementById('model-stats').innerText = `${this.splatCount.toLocaleString()} Gaussians | ${this.fps} FPS`;
            }
        }

        const bg = this.bgDark ? 0.07 : 0.95;
        gl.clearColor(bg, bg, bg, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        if (this.splatCount > 0) {
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            gl.disable(gl.DEPTH_TEST);

            gl.useProgram(this.program);

            const eye = [
                this.camera.target[0] + this.camera.radius * Math.sin(this.camera.phi) * Math.sin(this.camera.theta),
                this.camera.target[1] + this.camera.radius * Math.cos(this.camera.phi),
                this.camera.target[2] + this.camera.radius * Math.sin(this.camera.phi) * Math.cos(this.camera.theta)
            ];

            const view = this.createLookAtMatrix(eye, this.camera.target, [0, 1, 0]);
            const aspect = this.canvas.width / this.canvas.height;
            const proj = this.createPerspectiveMatrix(this.camera.fov, aspect, 0.1, 100.0);

            gl.uniformMatrix4fv(this.u_view, false, view);
            gl.uniformMatrix4fv(this.u_proj, false, proj);
            gl.uniform2f(this.u_viewport, this.canvas.width, this.canvas.height);

            gl.bindVertexArray(this.vao);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.splatCount);
            gl.bindVertexArray(null);
        }

        requestAnimationFrame((t) => this.renderLoop(t));
    }

    createLookAtMatrix(eye, center, up) {
        let z = [eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]];
        let len = Math.hypot(...z); z = z.map(v => v/len);
        let x = [up[1]*z[2] - up[2]*z[1], up[2]*z[0] - up[0]*z[2], up[0]*z[1] - up[1]*z[0]];
        len = Math.hypot(...x); x = x.map(v => v/len);
        let y = [z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2], z[0]*x[1] - z[1]*x[0]];
        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]),
            -(y[0]*eye[0] + y[1]*eye[1] + y[2]*eye[2]),
            -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]), 1
        ]);
    }

    createPerspectiveMatrix(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0
        ]);
    }
}

window.viewer = null;
window.addEventListener('DOMContentLoaded', () => {
    window.viewer = new GaussianSplatViewer();
});

window.loadSplatFromUrl = function(url, name) {
    if (window.viewer) {
        document.getElementById('model-name').innerText = name || '3D Gaussian Splat';
        window.viewer.loadSplat(url);
    }
};
