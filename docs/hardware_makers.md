# Hardware de los makers: rangos de cómputo, memoria y cámara

Investigación del 2026-10-01, fase 1 de 2 (la fase 2 elige modelos públicos para cada rango). Objetivo: saber qué hardware compra de verdad la gente que usaría EyeForge, para dimensionar el catálogo de modelos (`sim2chip/hardware.py`) por rangos reales y no por intuición.

**Límite de la fuente.** No hay cifras públicas de ventas por placa. La popularidad sale de indicios: guías de compra de 2026, placas soportadas por Edge Impulse y SenseCraft, ejemplos oficiales de Espressif y Raspberry Pi, y precios. Las cifras de rendimiento son de terceros con modelos parecidos a los nuestros, no medidas aquí.

## Rangos

| Rango | Placas típicas | Precio | Memoria para el modelo | Acelerador | Cámara habitual | Formato del modelo |
|---|---|---|---|---|---|---|
| R0. Micro sin acelerador | ESP32-CAM (AI-Thinker), XIAO ESP32-S3 Sense, ESP32-S3-EYE, Freenove S3 | 5-15 $ (EYE 45 $) | 512-520 KB SRAM + 4-8 MB PSRAM, 4-8 MB flash | ninguno; el S3 tiene instrucciones vectoriales (ESP-NN, unas 7x más rápido) | OV2640 2 MP (vídeo 640x480), a veces OV3660 o OV5640 | TFLite Micro, ESP-DL |
| R0b. Micro Cortex-M7 | Arduino Nicla Vision, OpenMV H7 | 65-80 $ | 1 MB SRAM, 2 MB flash (OpenMV H7 Plus añade 32 MB SDRAM) | ninguno | 2 MP | TFLite Micro (OpenMV, Edge Impulse) |
| R1. Micro con NPU | ESP32-P4 (P4-EYE, FireBeetle P4), Grove Vision AI V2 (Himax HX6538, Ethos-U55) | 16-40 $ | P4: hasta 32 MB PSRAM; Himax: unos 2,5 MB SRAM y 16 MB flash (sin verificar) | P4: extensiones de IA, unas 2x el S3; U55: NPU | P4: MIPI-CSI 2 MP (SC2336) con ISP; Himax: OV5647 | ESP-DL (P4); TFLite + compilador Vela (U55) |
| R2. Linux pequeño con NPU | Sipeed MaixCAM (SG2002), Luckfox Pico (RV1106), Raspberry Pi AI Camera (IMX500) | 35-70 $ | 64-256 MB RAM; IMX500: 8 MB para modelo y memoria de trabajo juntos | 0,5-1 TOPS; IMX500: procesador en el propio sensor | 4-5 MP; IMX500 12 MP | MaixCAM: cvimodel; Luckfox: RKNN; IMX500: empaquetador de Sony |
| R3. Placa Linux | Raspberry Pi 5 (sola o con AI HAT+), Pi Zero 2 W, Coral USB (antiguo) | Pi 5 60-80 $; con HAT unos 150 $ | 2-16 GB RAM | sin HAT, CPU; AI HAT+ 13 o 26 TOPS (Hailo-8L/8); AI HAT+ 2 40 TOPS INT4 (Hailo-10H); Coral 4 TOPS | Camera Module 3 (IMX708 12 MP, autofoco), v2 (IMX219 8 MP), OV5647 5 MP | TFLite/ONNX en CPU; HEF de Hailo; Edge TPU |
| R4. GPU de borde | Jetson Orin Nano Super | 249 $ | 8 GB | 67 TOPS INT8, CUDA | CSI o USB | TensorRT, ONNX |
| R5. Sin aparato | móvil o PC del usuario | ya lo tiene | sobrada | variable | móvil o webcam | TFLite, ONNX |

## Tiempos medidos por terceros (por foto)

| Placa | Modelo | Entrada | Tiempo |
|---|---|---|---|
| ESP32-S3 (XIAO) | MobileNetV2 0,35 int8 | 96x96, gris | 145-160 ms (80-120 ms en detección de personas) |
| ESP32-S3 | detector de peatones de Espressif | 224x224 | 130 ms en total (118 ms el modelo) |
| ESP32-P4 | el mismo detector | 224x224 | 60 ms en total (56 ms el modelo) |
| ESP32-S3 | YOLO11n | 320x320 | 7,7 s (inútil) |
| ESP32-P4 | YOLO11n | 320-640 | 0,55-2,5 s |
| Grove Vision AI V2 | YOLOv8n recortado | 192x192 | 33 ms (30 fps, 0,35 W) |
| IMX500 | SSD MobileNetV2 FPN Lite | 320x320 | en tiempo real (la cámara lo trae de serie) |

## Conclusiones para el catálogo

1. **El salto grande no es de potencia, es de formato.** Cada rango con acelerador usa su propio compilador (Vela, RKNN, cvimodel, HEF de Hailo, empaquetador del IMX500, TensorRT). Hoy exportamos TFLite int8, que sirve en R0, R0b, R3 en CPU y R5. Cada formato nuevo es trabajo de exportación aparte, y algunos (Hailo, IMX500) piden registro o herramientas cerradas del fabricante.
2. **En los micros, la RAM decide la resolución y la flash decide el tamaño del modelo.** Con 512 KB de SRAM y PSRAM lenta, 96-160 px es el rango realista en R0. El P4 y el U55 permiten 192-224 px en tiempos útiles.
3. **La cámara casi nunca es el cuello de botella en número de píxeles.** Hasta la OV2640 da 640x480, más que cualquier entrada de modelo de R0-R2. Lo que cambia entre cámaras es la nitidez, el ruido y la luz baja (eso ya lo modela `sharpness` y `look.py`). La resolución extra del sensor solo sirve si el aparato puede trocear la foto en ventanas (útil para objetos pequeños al contar), y eso cuesta cómputo.
4. **El hueco más poblado y barato es R0 (ESP32-S3 con OV2640).** Es nuestro caso de demo y el de la mayoría de makers. R1 (P4 y Grove Vision AI V2) es el siguiente escalón natural: precio de micro y 2-4 veces la velocidad.
5. **El perfil `jetson_nano` está anticuado.** La Jetson Nano original está descatalogada; la actual es la Orin Nano Super. Y faltan perfiles para R1, R2 y la Raspberry Pi 5, que es probablemente la placa Linux más común entre makers.

## Móviles (público de 18 a 30 años)

Ventas: en el segundo trimestre de 2026, los 10 móviles más vendidos del mundo fueron iPhone 17, 17 Pro Max, 17 Pro, 17e y 16, y Galaxy S26 Ultra, S26, A17 5G, A17 4G y A07 (Counterpoint). En Europa, Samsung y Apple tienen un 34 % cada uno y Xiaomi un 15 %. Xiaomi es más fuerte en España e Italia. No hay datos públicos fiables por edad; una fuente débil da un 31 % de iPhone en la generación Z global.

| Gama | Ejemplos | Chip | RAM | Acelerador | Cámara principal |
|---|---|---|---|---|---|
| Baja (100-200 $) | Galaxy A07, A17, Redmi Note 14 | Helio G99, Exynos 1330, Dimensity 6300/7025 | 4-8 GB | NPU débil o ninguna | 50 MP con mucho procesado |
| Media (300-450 $) | Galaxy A56, Redmi Note 15 Pro, Pixel "a" | Exynos 1580 (NPU de 14,7 TOPS), Dimensity 7400 | 8-12 GB | NPU media | 50-200 MP |
| Alta (800 $ o más) | iPhone 17 y 17 Pro, Galaxy S26, Pixel 10 | A19 Pro, Snapdragon 8 Elite Gen 5, Tensor G5 | 8-16 GB | NPU de gama alta | 48-200 MP, varias lentes |

Conclusiones:
1. **Potencia y memoria sobran en cualquier gama.** Un MobileNet de 224 px en int8 tardaba unos 13 ms en la CPU de un Pixel 3 (2018). Un móvil barato de hoy está en ese orden, así que en el móvil caben modelos mucho mayores que los nuestros (más resolución, ventanas para objetos pequeños, más clases). El límite real es el tamaño de la descarga y la batería si se usa en continuo.
2. **El límite es dónde corre el modelo.** Nuestra app es una web (PWA), y en el navegador no se puede usar la NPU: corre en CPU (WebAssembly) o en GPU (WebGPU, según navegador y versión). Para la NPU haría falta una app nativa (Core ML en iPhone, LiteRT con delegado del fabricante en Android). Para nuestros tamaños de modelo, la CPU o la GPU del navegador bastan.
3. **La cámara del móvil es otro mundo frente a la OV2640.** Sobra resolución, y la imagen sale muy procesada: HDR, enfoque artificial, reducción de ruido y retoques. El modelo tiene que entrenarse con el aspecto de la cámara del usuario (`look.py` ya lo mide a partir de sus fotos), no con el de la ESP32.
4. **Propuesta:** un perfil "móvil" con entrada de 224-320 px, un modelo más ancho y exportación para el navegador. Sustituiría al perfil "generic" actual, que ya hace ese papel sin nombrarlo.

## Fase 2: modelos públicos por rango

Criterios: pesos preentrenados públicos, licencia compatible con vender licencias comerciales aparte (Apache o MIT; ver nota sobre Ultralytics), conversión a TFLite int8 probada por alguien y, si es posible, disponible en Keras (nuestro framework) para no convertir. Tamaño int8 aproximado = número de parámetros en bytes.

### Columnas (tronco preentrenado, sirve para clasificar y como base del conteo)

| Modelo | Parámetros | ImageNet top-1 | Dónde está | Licencia | int8 | Rango |
|---|---|---|---|---|---|---|
| MobileNetV2 0,35-1,4 (actual) | 0,4-6,1 M | 60-75 % | Keras, 5 resoluciones de 96 a 224 | Apache | muy bueno (ReLU6) | R0-R5 |
| MobileNetV3 Small/Large (variante "minimalistic") | 1,5-5,4 M | 61-75 % | Keras | Apache | bueno solo en la variante minimalistic (sin hard-swish) | R0-R3 |
| MCUNet (256 kB / 320 kB de SRAM) | menos de 1 MB de flash | 61-68 % (int8) | PyTorch + TFLite int8 ya hecho | MIT | nativo | R0 |
| MobileNetV4 Conv Small/Medium | 3,8 / 9,7 M | 74 / 80 % | timm (PyTorch); hay TFLite convertidos por terceros | Apache | bueno (solo ReLU), sin probar por nosotros | R1-R5 |
| EfficientNet-Lite0-4 | 4,7-13 M | 75-81 % | TF Hub / Kaggle | Apache | diseñado para int8 | R2-R5 |
| EfficientNetV2-B0 / ConvNeXt-Tiny | 7 / 28 M | 79 / 82 % | Keras | Apache | regular (swish, GELU); mejor en float16 | R4-R5 (móvil, PC, Jetson) |

### Detectores pequeños (alternativa a nuestra cabeza de puntos para contar)

| Modelo | Parámetros | COCO mAP | Licencia | Notas |
|---|---|---|---|---|
| Nuestra cabeza (puntos + densidad sobre MobileNetV2 cortado) | menos de 1 M | no aplica | propia | igual que FOMO de Edge Impulse; única opción que cabe en R0 |
| PicoDet-S / M | 1,0 / 2,2 M | 30,6 / 34,3 % | Apache (PaddleDetection, activo) | el mejor equilibrio pequeño; exporta a ONNX; Paddle, no Keras |
| NanoDet-Plus | 1,2 M | hasta 34 % | Apache (sin cambios desde 2024) | 980 KB en int8, 10 ms en un móvil |
| YOLOX-Nano | 0,9 M | 25,8 % | Apache | más flojo que PicoDet |
| Swift-YOLO (Seeed, SSCMA) | pequeño, 96-320 px | variable | **repo sin licencia declarada** | hecho para Grove Vision AI V2 y XIAO S3; no usar hasta aclararla |
| YOLO26n / YOLO11n (Ultralytics) | 2,4 M | 40 % | **AGPL-3.0** | es el mejor, pero los modelos que entregáramos al usuario quedarían bajo AGPL, lo que choca con vender licencias comerciales; sin licencia Enterprise no lo usaría |
| Detectores en el ESP32-S3 | | | | YOLO11n a 320 px tarda 7,7 s en el S3: los detectores tipo YOLO no caben en R0 |

Conteo de multitudes con densidad: MobileCount (0,9 M) no tiene licencia, así que no se puede usar. Nuestra salida de densidad cubre ese papel.

### Recomendación por rango

| Rango | Clasificar | Contar |
|---|---|---|
| R0 (ESP32-CAM, S3, Nicla) | MobileNetV2 0,35-0,5 a 96-128 px (actual). Probar MCUNet y MobileNetV3 Small minimalistic. | Cabeza propia (puntos + densidad). Es lo único que cabe. |
| R1 (ESP32-P4, Grove Vision AI V2) | MobileNetV2 0,5-1,0 a 160-224 px | Cabeza propia a más resolución; probar PicoDet-S a 192-320 px |
| R2-R3 (Pi 5, MaixCAM, Luckfox, Hailo) | MobileNetV2 1,0, MobileNetV4 o EfficientNet-Lite a 224 px | PicoDet-S/M o NanoDet-Plus a 320-416 px, más trocear la foto si los objetos son pequeños |
| R4-R5 (Jetson, móvil, PC, navegador) | EfficientNetV2-B0 o ConvNeXt-Tiny | PicoDet-M/L a 416-640 px, troceando la foto |

### Resultado de la prueba de clasificación en micros (2026-10-02)

`experiments/classify_backbones.py`, receta de pocas fotos (tronco congelado, última capa), modelo int8 del chip, 10 repeticiones con las mismas particiones. Césped: 3 clases, la mitad de las 69 fotos para entrenar. MIO-TCD: coche, autobús, moto y peatón, 15 fotos por clase para entrenar y 100 para el test. Entre paréntesis, diferencia con el actual y repeticiones ganadas/perdidas.

| Tronco | Césped | MIO-TCD | Tamaño | RAM de trabajo |
|---|---|---|---|---|
| MobileNetV2 0,35 a 96 px (actual) | 73,7 % | 70,7 % | 616 KB | 135 KB |
| MobileNetV2 0,35 a 128 px | 77,7 % (+4,0, 6/1) | 71,0 % (+0,3, 6/4) | 616 KB | 240 KB |
| MobileNetV2 0,35 a 160 px | 75,7 % (+2,0, 7/2) | 73,3 % (+2,7, 7/3) | 616 KB | 375 KB |
| MobileNetV2 0,5 a 96 px | 79,1 % (+5,4, 8/1) | 69,8 % (-0,8, 3/6) | 957 KB | 135 KB |
| MobileNetV2 0,5 a 128 px | 74,6 % (+0,9, 6/3) | 71,2 % (+0,6, 6/4) | 957 KB | 240 KB |
| MobileNetV2 1,0 a 96 px | 72,9 % (-0,9, 4/5) | 71,3 % (+0,7, 6/4) | 2,6 MB | 270 KB |
| MobileNetV2 1,0 a 160 px | 79,1 % (+5,4, 8/1) | 69,9 % (-0,8, 5/5) | 2,6 MB | 750 KB |
| MobileNetV3 Small minimalistic a 96 px | 68,9 % (-4,9, 3/6) | 61,9 % (-8,8, 0/10) | 605 KB | 63 KB |
| MobileNetV3 Small minimalistic a 128 px | 77,1 % (+3,4, 7/2) | 63,4 % (-7,3, 1/9) | 605 KB | 112 KB |
| MobileNetV3 Small a 96 px | 73,4 % (-0,3, 6/3) | 63,3 % (-7,3, 0/10) | 1,2 MB | 72 KB |

Conclusiones:
1. **MobileNetV3 Small, descartado.** Pierde 7-9 puntos en el tráfico en casi todas las repeticiones. La versión completa, además, no arranca en int8 con XNNPACK, el motor de CPU de la web.
2. **El ancho y la resolución mueven poco con esta receta**, 2-5 puntos, y lo que gana en un dataset no gana en el otro. El 0,5 a 96 px, el mejor en el césped, pierde en el tráfico.
3. **La única mejora en los dos es MobileNetV2 0,35 a 160 px** (+2,0 y +2,7, con la misma flash). Cuesta casi 3 veces la RAM de trabajo (375 KB, que en la ESP32-CAM va a la PSRAM, más lenta) y unas 2,8 veces el tiempo por foto (de 80-160 ms a unos 250-450 ms en el S3, estimado, sin medir). Vale para detectar presencia cada cierto tiempo, no para vídeo.
4. **El catálogo de micros se queda con MobileNetV2.** La resolución se elige por el tiempo por foto que acepte el usuario. Las mejoras grandes no vendrán de cambiar de tronco en el micro.

### Resultado de la prueba de clasificación en móvil (2026-10-02)

Mismo script y particiones (`classify_backbones.py all 10 phone`), todos a 224 px. Se mide el modelo int8 y también el float, para separar el tronco de su conversión. Entre paréntesis, diferencia con el actual (MobileNetV2 1,0 int8, perfil "generic") y repeticiones ganadas/perdidas.

| Tronco | Césped int8 | Césped float | MIO-TCD int8 | MIO-TCD float | Tamaño int8 |
|---|---|---|---|---|---|
| MobileNetV2 1,0 (actual) | 75,4 % | 74,6 % | 69,7 % | 70,9 % | 2,6 MB |
| MobileNetV3 Large | 55,4 % (-20,0, 0/9) | 80,9 % (+5,4, 7/2) | 52,9 % (-16,8, 0/10) | 71,5 % (+1,8, 7/3) | 3,4 MB |
| **EfficientNetV2-B0** | **84,6 % (+9,1, 8/1)** | 82,6 % (+7,1, 7/2) | **73,0 % (+3,3, 8/2)** | 74,4 % (+4,7, 9/0) | 7,0 MB |
| ConvNeXt-Tiny | 32,3 % (-43,1, 0/10) | 80,9 % (+5,4, 7/1) | 49,9 % (-19,8, 0/10) | 71,2 % (+1,5, 9/1) | 28 MB |

Conclusiones:
1. **EfficientNetV2-B0 es el candidato para móvil**: gana en los dos datasets, en int8 y en float, y el int8 no pierde nada. Licencia Apache 2.0 (Keras).
2. **MobileNetV3 Large y ConvNeXt-Tiny se rompen en int8** con la cuantización tras el entrenamiento (hard-swish y LayerNorm). En float son buenos troncos, pero no mejores que EfficientNetV2-B0, y ConvNeXt pesa 4 veces más. Descartados.
3. Velocidad medida en un Pixel 7a (abajo): EfficientNetV2-B0 int8 tarda 64 ms por foto con LiteRT.js, de sobra para la prueba en vivo.

### Velocidad en el navegador del móvil (2026-10-02, Pixel 7a, Chrome 154)

`experiments/phone_speed/` (`export.py` genera los modelos, `vite.config.mjs` sirve la página y guarda los resultados en `runs/bench/phone_speed/`). Mediana de ms por foto a 224 px, contando la lectura de la salida, como la prueba en vivo. WebAssembly en un hilo, como la app (sin aislamiento entre orígenes).

| Modelo | LiteRT.js WASM | LiteRT.js WebGPU | ORT Web WASM | ORT Web WebGPU |
|---|---|---|---|---|
| MobileNetV2 1,0 int8 (actual) | 28 | 32 | 80 | 426 |
| MobileNetV2 1,0 fp16 | 23 | 30 | 57 | 30 |
| EfficientNetV2-B0 int8 | **64** | no compila | 126 | 1972 |
| EfficientNetV2-B0 fp16 | 56 | no compila | 112 | 36 |

Conclusiones:
1. **LiteRT.js en WASM (XNNPACK) es lo más rápido en CPU**, 2 veces más que ONNX Runtime Web. En WebGPU no gana a la CPU en modelos de este tamaño.
2. **EfficientNetV2-B0 no compila en LiteRT.js WebGPU** (error sin detalle en `litert_compiled_model_next.h`). La app ya cae a WASM si WebGPU falla, así que funcionaría tal cual, a 64 ms.
3. **ONNX Runtime Web no aporta nada aquí**: solo gana en WebGPU con fp16 (36 ms), y en int8 en WebGPU es inservible. No merece cambiar de runtime.
4. Decisión propuesta: perfil "generic" (móvil) con EfficientNetV2-B0 int8 a 224 px en LiteRT.js, con WASM directo para no perder tiempo intentando WebGPU. Falta medir un móvil de gama baja (estimado, sin medir: unas 3 veces más lento, unos 200 ms, aceptable para clasificar).

### Qué mediría primero (antes de tocar el catálogo)

1. **Clasificar en R0-R1:** MobileNetV2 0,35 frente a 0,5 y 1,0, y a 96 frente a 160 px, contra MobileNetV3 Small minimalistic. Todo está en Keras, así que la prueba es barata. Pruebas: césped y un dataset abierto, con tamaño y RAM medidos en TFLite.
2. **Contar en R2 o más:** PicoDet-S preentrenado en COCO frente a nuestra cabeza, en los gansos de FSC-147. Nuestra cabeza se estancó en un F1 de 0,60-0,65 y su cuello y su cabeza no traen nada preentrenado; un detector preentrenado podría pasar ese techo. Es la prueba con más potencial y también la más cara (Paddle u ONNX, no Keras).
3. Después de medir: perfiles nuevos (P4, Pi 5, móvil), quitar `jetson_nano` y separar la elección de resolución y ancho según la tarea.

## Fuentes

- [MCUNet (MIT Han Lab)](https://github.com/mit-han-lab/mcunet), [PP-PicoDet](https://arxiv.org/pdf/2111.00902), [NanoDet](https://github.com/RangiLyu/nanodet), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [MobileNetV4 TFLite convertidos (Hugging Face)](https://huggingface.co/collections/byoussef/mobilenetv4-conv-pretrained-tflite-models-66726c945040fc61f5a50b63), [timm](https://github.com/huggingface/pytorch-image-models)
- [SSCMA model zoo (Seeed)](https://github.com/seeed-studio/sscma-model-zoo), [Ultralytics YOLO26](https://www.ultralytics.com/yolo/yolo26), [export a LiteRT](https://docs.ultralytics.com/integrations/litert)
- [MobileCount](https://github.com/ChenyuGAO-CS/MobileCount)
- Licencias y actividad comprobadas en la API de GitHub el 2026-10-01.

- [iPhone 17 Global Best-selling Smartphone in Q2 2026 (Counterpoint)](https://counterpointresearch.com/en/insights/iphone-17-global-best-selling-smartphone-in-q2-2026)
- [Samsung ties Apple in Europe (SamMobile)](https://www.sammobile.com/news/samsung-ties-apple-top-spot-europe-smartphone-market/)
- [iPhone vs. Android Statistics 2026 (SQ Magazine)](https://sqmagazine.co.uk/iphone-vs-android-statistics/)
- [Galaxy A17 (GSMArena)](https://www.gsmarena.com/samsung_galaxy_a17_5g-14041.php), [Galaxy A07 (techspecs)](https://www.techspecs.info/samsung-galaxy-a07/), [Galaxy A56 (GSMArena)](https://www.gsmarena.com/samsung_galaxy_a36_and_a56_hands_on-review-2804p4.php), [Redmi Note 15 Pro (91mobiles)](https://www.91mobiles.com/xiaomi-redmi-note-15-pro-price-in-india)
- [Mobile SoCs ranking (AI-Benchmark)](https://ai-benchmark.com/ranking_processors)
- [MLPerf Mobile Inference Benchmark](https://arxiv.org/pdf/2012.02328)

- [Best TinyML and Edge AI Development Kits in 2026 (Microcontrollers Lab)](https://microcontrollerslab.com/best-tinyml-edge-ai-development-kits-buying-guide/)
- [ESP32 Camera Guide 2026 (Seeed Studio)](https://www.seeedstudio.com/blog/2026/05/06/esp32-camera-guide/)
- [Best ESP32 Camera Boards for Beginners and Makers (Component Advisor)](https://componentadvisor.com/best-esp32-camera-boards-for-beginners/)
- [Introducing ESP32-P4-EYE (Espressif)](https://developer.espressif.com/blog/2025/05/introducing-p4-eye/)
- [esp-dl pedestrian_detect y modelos (Espressif)](https://github.com/espressif/esp-dl/tree/master/models/pedestrian_detect)
- [Grove Vision AI V2 (CNX Software)](https://www.cnx-software.com/2024/01/19/16-grove-vision-ai-v2-module-features-wiseeye2-hx6538-arm-cortex-m55-ethos-u55-ai-microcontroller/)
- [Getting Started with the Grove Vision AI V2 (Core Electronics)](https://core-electronics.com.au/guides/getting-started-with-the-grove-vision-ai-v2-power-efficient-object-detection/)
- [Raspberry Pi AI Camera, model zoo IMX500](https://github.com/raspberrypi/imx500-models)
- [Best Edge AI Hardware 2026 (OpenELAB)](https://openelab.io/blogs/learn/best-edge-ai-hardware-2026-raspberry-pi-jetson-rk3588-ai-camera)
- [Jetson Orin Nano Super vs Raspberry Pi 5 + Hailo-8L (myaihardware)](https://www.myaihardware.com/compare-article/jetson-orin-nano-super-vs-raspberry-pi-5-hailo)
- [Luckfox Pico](https://www.luckfox.com/Luckfox-Pico) y [Sipeed MaixCAM](https://spotpear.com/shop/Sipeed-MaixCAM-SG2002-RISC-V-AI-YOLO-Camera-Visual-Audio-WIFI6-Linux.html)
- [Benchmarking tiny image classification models on ESP32 (TNU)](https://jst.tnu.edu.vn/jst/article/viewFile/14341/pdf)
- [ESP32 camera modules compared (espboards.dev)](https://www.espboards.dev/blog/esp32-camera-modules-compared/)
- [Raspberry Pi Camera v2 vs v3 vs HQ (Zbotic)](https://zbotic.in/raspberry-pi-camera-v2-vs-v3-vs-hq-which-one-to-choose/)
