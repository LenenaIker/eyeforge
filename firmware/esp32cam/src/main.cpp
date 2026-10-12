// EyeForge runner for the AI-Thinker ESP32-CAM: reads the model from the "model" partition
// (model.bin from sim2chip's export stage), runs it on the centre square of each camera frame
// and prints one JSON line per frame on the serial port: the label of a classification model,
// or the count per class of a count model (with where each object is), plus the square the
// model saw as a small JPEG. The ROM and bootloader talk at 115200 baud; this program switches
// to APP_BAUD first thing, so frames and photos fit through the cable.
//
// Lines it prints (the web app reads them, web/src/flash/board.ts):
//   {"event":"ready","firmware":"1.1.0","crc":"1a2b3c4d","task":"classify","input":96,...,"labels":[...]}
//   {"label":"grass","score":0.91,"ms":1210,"scores":{...},"jpeg":"<base64>"}
//   {"ms":1300,"counts":{"bottle":3},"points":[[0.42,0.25,"bottle"],...],"jpeg":"<base64>"}
//   {"event":"photo","w":800,"h":600,"bytes":41234,"capture_ms":610,"jpeg":"<base64>"}
//   {"event":"photo_sent","bytes":41234,"send_ms":720}   (after the photo, for measuring)
//   {"event":"error","message":"...","firmware":"1.1.0"}
// Points are 0-1 on the square the model saw (the centre square of the frame).
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
#include "driver/uart.h"
#include "driver/uart_vfs.h"
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

#define FIRMWARE_VERSION "1.1.0"  // tools/copy_web.sh copies it into the web's manifest.json

constexpr int APP_BAUD = 921600;
constexpr size_t ARENA = 512 * 1024;  // PSRAM: the internal RAM has no block this big
constexpr int MAX_CLASSES = 16;
constexpr int MAX_POINTS = 64;
constexpr uint32_t HEADER_MIN = 24;
constexpr int CROP_QUALITY = 80;   // JPEG quality of the square the model saw
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

// The console UART through the driver (so commands can be read) at APP_BAUD.
static void initSerial() {
    fflush(stdout);
    if (uart_driver_install(UART_NUM_0, 1024, 0, 0, nullptr, 0) != ESP_OK) return;  // stays at boot speed
    uart_vfs_dev_use_driver(UART_NUM_0);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
    uart_set_baudrate(UART_NUM_0, APP_BAUD);
}

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

// Inference: QVGA RGB565 (320x240; the centre 240x240 is scaled to the model's side). A photo: JPEG
// from the sensor itself at the size asked for.
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

// RGB565 centre square to the model's input (RGB or grey), quantized with the tensor's own params.
// crop gets the same pixels as the camera gives them (RGB565, or grey for a grey model) for the JPEG.
static void toInput(const camera_fb_t* fb, TfLiteTensor* t, int side, int channels, uint8_t* crop) {
    const int square = std::min((int)fb->width, (int)fb->height);
    const int x0 = (fb->width - square) / 2, y0 = (fb->height - square) / 2;
    const float scale = t->params.scale;
    const int zero = t->params.zero_point;
    int8_t* d = t->data.int8;
    for (int y = 0; y < side; y++) {
        for (int x = 0; x < side; x++) {
            const int i = ((y0 + y * square / side) * fb->width + x0 + x * square / side) * 2;
            const uint16_t p = fb->buf[i] << 8 | fb->buf[i + 1];
            const int rgb[3] = {((p >> 11) & 31) * 255 / 31, ((p >> 5) & 63) * 255 / 63, (p & 31) * 255 / 31};
            if (channels == 1) {
                const int grey = (rgb[0] * 77 + rgb[1] * 150 + rgb[2] * 29) >> 8;
                *d++ = (int8_t)std::clamp((int)lroundf(grey / scale) + zero, -128, 127);
                *crop++ = (uint8_t)grey;
            } else {
                for (int k = 0; k < 3; k++) *d++ = (int8_t)std::clamp((int)lroundf(rgb[k] / scale) + zero, -128, 127);
                *crop++ = fb->buf[i];
                *crop++ = fb->buf[i + 1];
            }
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
    esp_camera_deinit();
    if (initCamera(PIXFORMAT_JPEG, size) == ESP_OK) {
        camera_fb_t* fb = nullptr;
        for (int i = 0; i < 4; i++) {  // the first frames after a restart are dark: let exposure settle
            if (fb) esp_camera_fb_return(fb);
            fb = esp_camera_fb_get();
        }
        if (fb) {
            const unsigned captureMs = msSince(t0);
            const int64_t t1 = esp_timer_get_time();
            printf("{\"event\":\"photo\",\"w\":%u,\"h\":%u,\"bytes\":%u,\"capture_ms\":%u,\"jpeg\":\"",
                   (unsigned)fb->width, (unsigned)fb->height, (unsigned)fb->len, captureMs);
            writeBase64(fb->buf, fb->len);
            printf("\"}\n");
            fflush(stdout);
            uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(10000));
            printf("{\"event\":\"photo_sent\",\"bytes\":%u,\"send_ms\":%u}\n", (unsigned)fb->len, msSince(t1));
            esp_camera_fb_return(fb);
        } else {
            printf("{\"event\":\"error\",\"message\":\"photo capture failed\"}\n");
        }
        esp_camera_deinit();
    } else {
        printf("{\"event\":\"error\",\"message\":\"camera does not take that photo size\"}\n");
    }
    fflush(stdout);
    if (initCamera() != ESP_OK) halt("camera init failed");
    for (int i = 0; i < 3; i++) {  // dark frames again after the restart: not worth a result
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
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
        if (uart_read_bytes(UART_NUM_0, &c, 1, now < end ? end - now : 0) != 1) return nullptr;
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
    const size_t cropBytes = (size_t)model.side * model.side * (channels == 1 ? 1 : 2);
    uint8_t* crop = (uint8_t*)heap_caps_malloc(cropBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!crop || (grid && !probs)) halt("out of memory");
    float threshold = 0.5f;  // sim2chip/counting.py THRESHOLD; the web sends the model's own

    printf("{\"event\":\"ready\",\"firmware\":\"" FIRMWARE_VERSION "\",\"crc\":\"%08lx\",\"task\":\"%s\",\"input\":%d,"
           "\"channels\":%d,\"grid\":%d,\"arena\":%u,\"labels\":[",
           (unsigned long)model.crc, model.task == 0 ? "classify" : "count", model.side, channels, grid,
           (unsigned)interpreter.arena_used_bytes());
    for (int c = 0; c < model.classes; c++) printf("%s\"%s\"", c ? "," : "", model.labels[c]);
    printf("]}\n");
    fflush(stdout);

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            printf("{\"event\":\"error\",\"message\":\"capture failed\"}\n");
            fflush(stdout);
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }
        toInput(fb, input, model.side, channels, crop);
        esp_camera_fb_return(fb);
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
        if (fmt2jpg(crop, cropBytes, model.side, model.side, channels == 1 ? PIXFORMAT_GRAYSCALE : PIXFORMAT_RGB565,
                    CROP_QUALITY, &jpg, &jpgLen)) {
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
