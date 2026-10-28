// EyeForge runner for the AI-Thinker ESP32-CAM and the Seeed XIAO ESP32S3 Sense: reads the model from the "model" partition
// (model.bin from sim2chip's export stage), runs it on each whole camera frame stretched to the
// model's square input (as the training photos are, sim2chip/stages/ingest.py)
// and prints one JSON line per frame on the serial port: the label of a classification model,
// or the count per class of a count model (with where each object is), plus the picture the
// model saw as a small JPEG. On the ESP32-CAM the ROM and bootloader talk at 115200 baud through
// its USB to serial chip; this program switches to APP_BAUD first thing, so frames and photos fit
// through the cable. The S3 talks over its own USB (USB Serial/JTAG), where the baud rate means nothing.
// Built with EYEFORGE_TEST_FRAME (env qemu_*, tools/qemu_test.sh), it runs in Espressif's QEMU: no
// camera, every frame is the fixed one of test_frame/test_frame.h.
//
// Lines it prints (the web app reads them, web/src/flash/board.ts); the ready line again every READY_EVERY frames:
//   {"event":"ready","firmware":"1.3.0","crc":"1a2b3c4d","task":"classify","input":96,"frame":[320,240],...,"labels":[...]}
//   {"label":"grass","score":0.91,"ms":1210,"scores":{...},"jpeg":"<base64>"}
//   {"ms":1300,"counts":{"bottle":3},"points":[[0.42,0.25,"bottle"],...],"jpeg":"<base64>"}
//   {"event":"photo","w":800,"h":600,"bytes":41234,"capture_ms":610,"jpeg":"<base64>"}
//   {"event":"photo_sent","bytes":41234,"send_ms":720}   (after the photo, for measuring)
//   {"event":"error","message":"...","firmware":"1.3.0"}
// Points are 0-1 on the picture the model saw, which is the whole frame stretched to a square, so they
// are also 0-1 on the whole frame and on a photo from "capture" (same field of view). Up to 1.1.0 the
// model saw the centre square of the frame instead, and the ready line had no "frame".
//
// Commands it reads, one per line: "capture [svga|xga|uxga|qxga]" sends one photo of the whole
// frame at that size (default uxga), then goes back to inference; "threshold 0.45" sets the
// probability a count cell needs to be an object (the model's own threshold, from its report).
//
// model.bin layout, little endian (sim2chip/stages/export.py, model_blob):
//   0  "EFM1"   4  header size   8  model size   12  CRC-32 of the model (zlib's)
//   16 input side (u16)   18 classes (u16)   20 task (u8: 0 classify, 1 count)   21-23 reserved
//   24 labels, each NUL-terminated, padded to the header size; the .tflite follows.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "sdkconfig.h"
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#else
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#endif
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "img_converters.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "camera_pins.h"
#ifdef EYEFORGE_TEST_FRAME
#include "test_frame.h"  // TEST_FRAME_W, TEST_FRAME_H, test_frame[]: RGB565, big endian as the camera's
#endif

#define FIRMWARE_VERSION "1.3.0"  // tools/copy_web.sh copies it into the web's manifest.json

// 921600 lost bytes in the ESP32-CAM-MB's CH340 (2026-10-28: the ready line came without the same 24 bytes every
// time, on a computer and on a phone); a frame and its JPEG are about 4 KB a second, a 1600x1200 photo 75 KB.
constexpr int APP_BAUD = 460800;
constexpr int READY_EVERY = 5;  // frames between repeats of the ready line, for a reader that missed the first
constexpr size_t ARENA = 512 * 1024;  // PSRAM: the internal RAM has no block this big
constexpr int MAX_CLASSES = 16;
constexpr int MAX_POINTS = 64;
constexpr uint32_t HEADER_MIN = 24;
constexpr int VIEW_QUALITY = 80;   // JPEG quality of the picture the model saw
constexpr int PHOTO_QUALITY = 12;  // esp32-camera's scale: lower is better
constexpr int IDLE_MS = 100;       // pause between frames, spent listening for commands

struct Model {
    const uint8_t* tflite;
    uint32_t crc;
    int side;
    int classes;
    int task;
    const char* labels[MAX_CLASSES];
};

static void halt(const char* message) {
    printf("{\"event\":\"error\",\"message\":\"%s\",\"firmware\":\"" FIRMWARE_VERSION "\"}\n", message);
    fflush(stdout);
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}

static uint32_t u32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t u16(const uint8_t* p) { return p[0] | p[1] << 8; }
static unsigned msSince(int64_t t0) { return (unsigned)((esp_timer_get_time() - t0) / 1000); }

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
// The chip's own USB through the driver, so commands can be read. With no computer listening, a line
// that cannot go out is dropped after a short wait (the driver's), so the board runs on its own too.
static void initSerial() {
    fflush(stdout);
    usb_serial_jtag_driver_config_t c = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    c.tx_buffer_size = 4096;
    c.rx_buffer_size = 1024;
    if (usb_serial_jtag_driver_install(&c) != ESP_OK) return;
    usb_serial_jtag_vfs_use_driver();
}
static bool readByte(uint8_t* c, TickType_t wait) { return usb_serial_jtag_read_bytes(c, 1, wait) == 1; }
static void waitSent(int ms) { usb_serial_jtag_wait_tx_done(pdMS_TO_TICKS(ms)); }
#else
// The console UART through the driver (so commands can be read) at APP_BAUD.
static void initSerial() {
    fflush(stdout);
    if (uart_driver_install(UART_NUM_0, 1024, 0, 0, nullptr, 0) != ESP_OK) return;  // stays at boot speed
    uart_vfs_dev_use_driver(UART_NUM_0);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
    uart_set_baudrate(UART_NUM_0, APP_BAUD);
}
static bool readByte(uint8_t* c, TickType_t wait) { return uart_read_bytes(UART_NUM_0, c, 1, wait) == 1; }
static void waitSent(int ms) { uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(ms)); }
#endif

static Model loadModel() {
    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
    if (!part) halt("no model partition in the partition table");
    uint8_t head[HEADER_MIN];
    if (esp_partition_read(part, 0, head, sizeof head) != ESP_OK) halt("cannot read the model partition");
    if (memcmp(head, "EFM1", 4) != 0) halt("no model flashed: write model.bin to the model partition");
    const uint32_t headerSize = u32(head + 4), size = u32(head + 8);
    if (headerSize < HEADER_MIN || headerSize % 16 || headerSize + size > part->size) halt("model.bin header is corrupt");

    const void* mapped;
    esp_partition_mmap_handle_t handle;
    if (esp_partition_mmap(part, 0, headerSize + size, ESP_PARTITION_MMAP_DATA, &mapped, &handle) != ESP_OK)
        halt("cannot map the model partition");
    const uint8_t* base = (const uint8_t*)mapped;
    if (esp_rom_crc32_le(0, base + headerSize, size) != u32(base + 12)) halt("model CRC mismatch: flash model.bin again");

    Model m = {base + headerSize, u32(base + 12), u16(base + 16), u16(base + 18), base[20], {}};
    if (m.classes < 1 || m.classes > MAX_CLASSES) halt("model.bin header is corrupt");
    const char* label = (const char*)base + HEADER_MIN;
    for (int c = 0; c < m.classes; c++) {
        m.labels[c] = label;
        label += strlen(label) + 1;
        if (label > (const char*)base + headerSize) halt("model.bin header is corrupt");
    }
    return m;
}

// Inference: QVGA RGB565 (320x240, stretched whole to the model's side). A photo: JPEG from the sensor
// itself at the size asked for, the same field of view.
#ifdef EYEFORGE_TEST_FRAME
// The emulator has no camera: every frame is the fixed one, and photos are refused.
static esp_err_t initCamera(pixformat_t format = PIXFORMAT_RGB565, framesize_t size = FRAMESIZE_QVGA) {
    return format == PIXFORMAT_RGB565 ? ESP_OK : ESP_ERR_NOT_SUPPORTED;
}
static camera_fb_t* grabFrame() {
    static camera_fb_t fb = {};
    fb.buf = (uint8_t*)test_frame;
    fb.len = sizeof test_frame;
    fb.width = TEST_FRAME_W;
    fb.height = TEST_FRAME_H;
    fb.format = PIXFORMAT_RGB565;
    return &fb;
}
static void releaseFrame(camera_fb_t*) {}
static void stopCamera() {}
#else
static camera_fb_t* grabFrame() { return esp_camera_fb_get(); }
static void releaseFrame(camera_fb_t* fb) { esp_camera_fb_return(fb); }
static void stopCamera() { esp_camera_deinit(); }

static esp_err_t initCamera(pixformat_t format = PIXFORMAT_RGB565, framesize_t size = FRAMESIZE_QVGA) {
    camera_config_t c = {};
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer = LEDC_TIMER_0;
    c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
    c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM;
    c.pin_href = HREF_GPIO_NUM; c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
    c.xclk_freq_hz = 20000000;
    c.pixel_format = format;
    c.frame_size = size;
    c.jpeg_quality = PHOTO_QUALITY;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = format == PIXFORMAT_JPEG ? CAMERA_GRAB_WHEN_EMPTY : CAMERA_GRAB_LATEST;
    return esp_camera_init(&c);
}
#endif

// Streams bytes as base64 to stdout, in pieces (a photo is too big to encode in one buffer).
static void writeBase64(const uint8_t* d, size_t n) {
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char buf[1024];
    size_t k = 0;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = d[i] << 16 | (i + 1 < n ? d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        buf[k++] = T[v >> 18 & 63];
        buf[k++] = T[v >> 12 & 63];
        buf[k++] = i + 1 < n ? T[v >> 6 & 63] : '=';
        buf[k++] = i + 2 < n ? T[v & 63] : '=';
        if (k + 4 > sizeof buf) { fwrite(buf, 1, k, stdout); k = 0; }
    }
    fwrite(buf, 1, k, stdout);
}

// Pillow's bilinear resize (Image.resize(..., BILINEAR)), as ingest stretches the training photos and
// web/src/live/resize.ts the phone's frames: a triangle filter whose support grows with the reduction,
// horizontal pass then vertical, fixed-point weights (22 bits) and 8-bit rounding after each pass. The
// weights depend only on the sizes, so they are worked out once.
struct Taps {
    int in = 0, out = 0, ksize = 0;
    int* start = nullptr;
    int* size = nullptr;
    int32_t* w = nullptr;
};
constexpr int PREC_BITS = 22;

static void makeTaps(Taps& t, int in, int out) {
    if (t.in == in && t.out == out) return;
    free(t.start); free(t.size); free(t.w);
    const double scale = (double)in / out, fscale = std::max(scale, 1.0), support = fscale;
    t.in = in; t.out = out; t.ksize = (int)std::ceil(support) * 2 + 1;
    t.start = (int*)malloc(out * sizeof(int));
    t.size = (int*)malloc(out * sizeof(int));
    t.w = (int32_t*)malloc(out * t.ksize * sizeof(int32_t));
    if (!t.start || !t.size || !t.w) halt("out of memory");
    double w[64];
    for (int x = 0; x < out; x++) {
        const double center = (x + 0.5) * scale;
        const int lo = std::max((int)(center - support + 0.5), 0);
        const int n = std::min((int)(center + support + 0.5), in) - lo;
        double total = 0;
        for (int i = 0; i < n && i < 64; i++) {
            const double d = std::fabs((i + lo - center + 0.5) / fscale);
            w[i] = d < 1 ? 1 - d : 0;
            total += w[i];
        }
        for (int i = 0; i < t.ksize; i++)
            t.w[x * t.ksize + i] = i < n && total ? (int32_t)(w[i] / total * (1 << PREC_BITS) + 0.5) : 0;
        t.start[x] = lo;
        t.size[x] = std::min(n, t.ksize);
    }
}

static inline uint8_t clip8(int32_t v) { v >>= PREC_BITS; return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }

// The whole RGB565 frame stretched to the model's side (no crop, as ingest does with the training photos)
// into rgb (side x side x 3, R G B), then quantized into the model's input with the tensor's own params.
// view gets the same picture for the preview JPEG: B G R (esp32-camera's RGB888 order), or grey.
static void toInput(const camera_fb_t* fb, TfLiteTensor* t, int side, int channels, uint8_t* view) {
    static Taps tx, ty;
    static uint8_t* row = nullptr;   // one frame row as RGB888
    static uint8_t* tmp = nullptr;   // horizontal pass: side x frame height x 3
    static uint8_t* rgb = nullptr;   // vertical pass: side x side x 3
    static int tmpH = 0;
    const int w = fb->width, h = fb->height;
    makeTaps(tx, w, side);
    makeTaps(ty, h, side);
    if (!rgb) {
        row = (uint8_t*)heap_caps_malloc(1600 * 3, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        rgb = (uint8_t*)heap_caps_malloc(side * side * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    if (tmpH != h) {
        free(tmp);
        tmp = (uint8_t*)heap_caps_malloc((size_t)side * h * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        tmpH = h;
    }
    if (!row || !rgb || !tmp || w > 1600) halt("out of memory");
    const int32_t half = 1 << (PREC_BITS - 1);
    for (int y = 0; y < h; y++) {
        const uint8_t* src = fb->buf + (size_t)y * w * 2;
        for (int x = 0; x < w; x++) {
            const uint16_t p = src[2 * x] << 8 | src[2 * x + 1];
            row[3 * x] = ((p >> 11) & 31) * 255 / 31;
            row[3 * x + 1] = ((p >> 5) & 63) * 255 / 63;
            row[3 * x + 2] = (p & 31) * 255 / 31;
        }
        uint8_t* out = tmp + (size_t)y * side * 3;
        for (int x = 0; x < side; x++) {
            int32_t r = half, g = half, b = half;
            const uint8_t* s = row + tx.start[x] * 3;
            const int32_t* k = tx.w + x * tx.ksize;
            for (int i = 0; i < tx.size[x]; i++, s += 3) { r += s[0] * k[i]; g += s[1] * k[i]; b += s[2] * k[i]; }
            out[3 * x] = clip8(r); out[3 * x + 1] = clip8(g); out[3 * x + 2] = clip8(b);
        }
    }
    for (int y = 0; y < side; y++) {
        const int32_t* k = ty.w + y * ty.ksize;
        for (int x = 0; x < side; x++) {
            int32_t r = half, g = half, b = half;
            const uint8_t* s = tmp + ((size_t)ty.start[y] * side + x) * 3;
            for (int i = 0; i < ty.size[y]; i++, s += side * 3) { r += s[0] * k[i]; g += s[1] * k[i]; b += s[2] * k[i]; }
            uint8_t* o = rgb + (y * side + x) * 3;
            o[0] = clip8(r); o[1] = clip8(g); o[2] = clip8(b);
        }
    }
    const float scale = t->params.scale;
    const int zero = t->params.zero_point;
    int8_t* d = t->data.int8;
    for (int i = 0; i < side * side; i++) {
        const uint8_t* p = rgb + i * 3;
        if (channels == 1) {
            const int grey = (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
            *d++ = (int8_t)std::clamp((int)lroundf(grey / scale) + zero, -128, 127);
            *view++ = (uint8_t)grey;
        } else {
            for (int k = 0; k < 3; k++) *d++ = (int8_t)std::clamp((int)lroundf(p[k] / scale) + zero, -128, 127);
            *view++ = p[2]; *view++ = p[1]; *view++ = p[0];
        }
    }
}

// The count model's points grid to objects, as sim2chip/counting.py decode (and web/src/live/decode.ts):
// a cell is an object when its best non-background class reaches the threshold; touching object cells
// of the same class are one object, at their probability-weighted centre, 0-1 of the square.
struct Point { float x, y; int cls; };

static int decodePoints(const float* probs, int g, int n, float threshold, Point* out) {
    static int best[64 * 64];
    static uint8_t on[64 * 64], seen[64 * 64];
    static int todo[64 * 64];
    if (g > 64 || n < 2) return 0;
    for (int k = 0; k < g * g; k++) {
        int b = 1;
        for (int c = 2; c < n; c++) if (probs[k * n + c] > probs[k * n + b]) b = c;
        best[k] = b;
        on[k] = probs[k * n + b] >= threshold;
        seen[k] = 0;
    }
    int found = 0;
    for (int k = 0; k < g * g && found < MAX_POINTS; k++) {
        if (!on[k] || seen[k]) continue;
        const int cls = best[k];
        int top = 0;
        todo[top++] = k;
        seen[k] = 1;
        float w = 0, sx = 0, sy = 0;
        while (top) {
            const int q = todo[--top], a = q / g, b = q % g;
            const float p = probs[q * n + cls];
            w += p; sx += (b + 0.5f) * p; sy += (a + 0.5f) * p;
            const int next[4][2] = {{a + 1, b}, {a - 1, b}, {a, b + 1}, {a, b - 1}};
            for (const auto& [y, x] : next) {
                const int r = y * g + x;
                if (y >= 0 && y < g && x >= 0 && x < g && on[r] && !seen[r] && best[r] == cls) {
                    seen[r] = 1;
                    todo[top++] = r;
                }
            }
        }
        out[found++] = {sx / w / g, sy / w / g, cls};
    }
    return found;
}

// One photo of the whole frame at a bigger size, as JPEG, then back to inference.
static void sendPhoto(framesize_t size) {
    const int64_t t0 = esp_timer_get_time();
    stopCamera();
    if (initCamera(PIXFORMAT_JPEG, size) == ESP_OK) {
        camera_fb_t* fb = nullptr;
        for (int i = 0; i < 4; i++) {  // the first frames after a restart are dark: let exposure settle
            if (fb) releaseFrame(fb);
            fb = grabFrame();
        }
        if (fb) {
            const unsigned captureMs = msSince(t0);
            const int64_t t1 = esp_timer_get_time();
            printf("{\"event\":\"photo\",\"w\":%u,\"h\":%u,\"bytes\":%u,\"capture_ms\":%u,\"jpeg\":\"",
                   (unsigned)fb->width, (unsigned)fb->height, (unsigned)fb->len, captureMs);
            writeBase64(fb->buf, fb->len);
            printf("\"}\n");
            fflush(stdout);
            waitSent(10000);
            printf("{\"event\":\"photo_sent\",\"bytes\":%u,\"send_ms\":%u}\n", (unsigned)fb->len, msSince(t1));
            releaseFrame(fb);
        } else {
            printf("{\"event\":\"error\",\"message\":\"photo capture failed\"}\n");
        }
        stopCamera();
    } else {
        printf("{\"event\":\"error\",\"message\":\"camera does not take that photo size\"}\n");
    }
    fflush(stdout);
    if (initCamera() != ESP_OK) halt("camera init failed");
    for (int i = 0; i < 3; i++) {  // dark frames again after the restart: not worth a result
        camera_fb_t* fb = grabFrame();
        if (fb) releaseFrame(fb);
    }
}

// Reads what came on the serial line, waiting at most wait ticks; returns a whole line or nullptr.
static const char* readCommand(TickType_t wait) {
    static char line[64];
    static int len = 0;
    const TickType_t end = xTaskGetTickCount() + wait;
    while (true) {
        uint8_t c;
        const TickType_t now = xTaskGetTickCount();
        if (!readByte(&c, now < end ? end - now : 0)) return nullptr;
        if (c == '\n' || c == '\r') {
            if (!len) continue;
            line[len] = 0;
            len = 0;
            return line;
        }
        if (len < (int)sizeof line - 1) line[len++] = (char)c;
    }
}

static void runCommand(const char* cmd, float& threshold) {
    if (strncmp(cmd, "capture", 7) == 0) {
        const char* arg = cmd + 7;
        while (*arg == ' ') arg++;
        const framesize_t size = !strcmp(arg, "qxga") ? FRAMESIZE_QXGA : !strcmp(arg, "uxga") ? FRAMESIZE_UXGA
                               : !strcmp(arg, "xga") ? FRAMESIZE_XGA : !strcmp(arg, "svga") ? FRAMESIZE_SVGA
                               : FRAMESIZE_UXGA;
        sendPhoto(size);
    } else if (strncmp(cmd, "threshold ", 10) == 0) {
        const float v = strtof(cmd + 10, nullptr);
        if (v > 0 && v < 1) threshold = v;
    }
}

extern "C" void app_main() {
    initSerial();
    const Model model = loadModel();
    if (initCamera() != ESP_OK) halt("camera init failed");

    const tflite::Model* graph = tflite::GetModel(model.tflite);
    if (graph->version() != TFLITE_SCHEMA_VERSION) halt("model schema version mismatch");
    uint8_t* arena = (uint8_t*)heap_caps_malloc(ARENA, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) halt("no PSRAM for the tensor arena");
    // The ops of our MobileNetV2 students and count heads, plus the common ones of small vision models.
    static tflite::MicroMutableOpResolver<19> ops;
    ops.AddAdd(); ops.AddAveragePool2D(); ops.AddConcatenation(); ops.AddConv2D(); ops.AddDepthwiseConv2D();
    ops.AddDequantize(); ops.AddFullyConnected(); ops.AddHardSwish(); ops.AddLogistic(); ops.AddMaxPool2D();
    ops.AddMean(); ops.AddMul(); ops.AddPad(); ops.AddQuantize(); ops.AddRelu(); ops.AddRelu6();
    ops.AddReshape(); ops.AddSoftmax(); ops.AddSub();
    static tflite::MicroInterpreter interpreter(graph, ops, arena, ARENA);
    if (interpreter.AllocateTensors() != kTfLiteOk) halt("AllocateTensors failed: model too big or unsupported op");
    TfLiteTensor* input = interpreter.input(0);
    const int channels = input->dims->data[3];
    if (input->type != kTfLiteInt8) halt("the model must be int8");
    if (input->dims->data[1] != model.side || (channels != 1 && channels != 3)) halt("unexpected model input shape");
    // Classification: one output, a probability per label. Count: the labels start with the background,
    // and the density output (told apart by its channels, one per other class) sums to the count; the
    // points output (one channel per label) says where the objects are.
    const int wanted = model.task == 0 ? model.classes : model.classes - 1;
    TfLiteTensor* output = nullptr;
    TfLiteTensor* pointsOut = nullptr;
    for (size_t o = 0; o < interpreter.outputs_size(); o++) {
        TfLiteTensor* t = interpreter.output(o);
        if (t->type != kTfLiteInt8) continue;
        const int last = t->dims->data[t->dims->size - 1];
        if (last == wanted) output = t;
        else if (model.task == 1 && last == model.classes) pointsOut = t;
    }
    if (!output) halt("model outputs do not match its labels");
    const int grid = pointsOut ? (int)lroundf(sqrtf((float)(pointsOut->bytes / model.classes))) : 0;
    float* probs = grid ? (float*)heap_caps_malloc(pointsOut->bytes * sizeof(float), MALLOC_CAP_SPIRAM) : nullptr;
    const size_t viewBytes = (size_t)model.side * model.side * (channels == 1 ? 1 : 3);
    uint8_t* view = (uint8_t*)heap_caps_malloc(viewBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!view || (grid && !probs)) halt("out of memory");
    float threshold = 0.5f;  // sim2chip/counting.py THRESHOLD; the web sends the model's own

    // Said at start and again every READY_EVERY frames: a page that opened the port late, or lost bytes of the
    // line, still learns which firmware and model run here.
    auto sayReady = [&]() {
        printf("{\"event\":\"ready\",\"firmware\":\"" FIRMWARE_VERSION "\",\"crc\":\"%08lx\",\"task\":\"%s\",\"input\":%d,"
               "\"frame\":[320,240],\"channels\":%d,\"grid\":%d,\"arena\":%u,\"labels\":[",
               (unsigned long)model.crc, model.task == 0 ? "classify" : "count", model.side, channels, grid,
               (unsigned)interpreter.arena_used_bytes());
        for (int c = 0; c < model.classes; c++) printf("%s\"%s\"", c ? "," : "", model.labels[c]);
        printf("]}\n");
        fflush(stdout);
    };
    sayReady();

    for (unsigned frame = 1;; frame++) {
        if (frame % READY_EVERY == 0) sayReady();
        camera_fb_t* fb = grabFrame();
        if (!fb) {
            printf("{\"event\":\"error\",\"message\":\"capture failed\"}\n");
            fflush(stdout);
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }
        toInput(fb, input, model.side, channels, view);
        releaseFrame(fb);
        const int64_t t0 = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) halt("inference failed");
        const unsigned ms = msSince(t0);
        float values[MAX_CLASSES] = {};
        const float scale = output->params.scale;
        const int zero = output->params.zero_point;
        if (model.task == 0) {
            int best = 0;
            for (int c = 0; c < model.classes; c++) {
                values[c] = (output->data.int8[c] - zero) * scale;
                if (values[c] > values[best]) best = c;
            }
            printf("{\"label\":\"%s\",\"score\":%.2f,\"ms\":%u,\"scores\":{", model.labels[best], values[best], ms);
            for (int c = 0; c < model.classes; c++) printf("%s\"%s\":%.2f", c ? "," : "", model.labels[c], values[c]);
            printf("}");
        } else {
            const int cells = output->bytes / wanted;
            for (int i = 0; i < cells * wanted; i++) values[i % wanted] += (output->data.int8[i] - zero) * scale;
            printf("{\"ms\":%u,\"counts\":{", ms);
            for (int c = 0; c < wanted; c++) printf("%s\"%s\":%ld", c ? "," : "", model.labels[c + 1], lroundf(values[c]));
            printf("}");
            if (grid) {
                const float ps = pointsOut->params.scale;
                const int pz = pointsOut->params.zero_point;
                for (size_t i = 0; i < pointsOut->bytes; i++) probs[i] = (pointsOut->data.int8[i] - pz) * ps;
                Point pts[MAX_POINTS];
                const int n = decodePoints(probs, grid, model.classes, threshold, pts);
                printf(",\"points\":[");
                for (int i = 0; i < n; i++)
                    printf("%s[%.4f,%.4f,\"%s\"]", i ? "," : "", pts[i].x, pts[i].y, model.labels[pts[i].cls]);
                printf("]");
            }
        }
        uint8_t* jpg = nullptr;
        size_t jpgLen = 0;
        if (fmt2jpg(view, viewBytes, model.side, model.side, channels == 1 ? PIXFORMAT_GRAYSCALE : PIXFORMAT_RGB888,
                    VIEW_QUALITY, &jpg, &jpgLen)) {
            printf(",\"jpeg\":\"");
            writeBase64(jpg, jpgLen);
            printf("\"");
            free(jpg);
        }
        printf("}\n");
        fflush(stdout);
        // The pause between frames listens for commands, so a photo asked for starts at once.
        const TickType_t until = xTaskGetTickCount() + pdMS_TO_TICKS(IDLE_MS);
        while (true) {
            const TickType_t now = xTaskGetTickCount();
            const char* cmd = readCommand(now < until ? until - now : 0);
            if (!cmd) break;
            runCommand(cmd, threshold);
        }
    }
}
