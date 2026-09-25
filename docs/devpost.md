# EyeForge: textos de Devpost

Registrados al crear el proyecto en Devpost (2026-09-23). Se pueden editar hasta la entrega. Mantener las cifras honestas: el 58% -> 79% está medido en MIO-TCD (cámaras de tráfico), no en la ESP32-CAM.

**Name:** EyeForge (antes Sim2Chip; cambiar también en Devpost)

**Track:** Physical AI

**Tagline:** Describe what your camera should see. Get a tiny model ready to flash on a $10 chip.

**Description:**

EyeForge trains vision models for low-power hardware like the ESP32-CAM, starting from a plain description and a handful of photos instead of thousands of labeled images.

An interview with NVIDIA Nemotron (Nebius Token Factory) turns the user's needs into a dataset spec: classes, lighting, weather, camera height, and details like "motorbikes always carry a rider". NVIDIA Isaac Sim generates labeled synthetic scenes from that spec on Nebius AI Cloud GPUs, using open 3D models checked by a vision model. If the user has a few raw photos, a large model labels them automatically. The result is a quantized int8 model small enough to run on a microcontroller.

In our first experiment (vehicle classes on real traffic-camera images), 50 real photos per class alone reached 58% accuracy; adding our synthetic data raised it to 79%, with the same photos. Our demo customer is a privacy-first LoRa alarm for cycling races: cameras at road crossings detect vehicles on the chip, discard the image and send only a few bits.
