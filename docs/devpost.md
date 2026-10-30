# EyeForge: textos de Devpost

Registrados al crear el proyecto en Devpost (2026-09-23) y puestos al día para la entrega (2026-10-30). Mantener las cifras honestas: el 58% -> 77% está medido en MIO-TCD (cámaras de tráfico), no en la ESP32-CAM.

**Name:** EyeForge. La URL de Devpost conserva el primer nombre: https://devpost.com/software/sim2chip

**Video:** https://youtu.be/sEvSlciHGKQ

**Demo:** https://eyeforge.app

**Track:** Best Apps and Agents (usuario, 2026-10-27)

**Tagline:** Describe what your camera should see. Get a tiny model ready to run on your own device.

**Description:**

EyeForge is a pipeline that turns a conversation and a few of your photos into an on-device vision model, trained with synthetic images rendered in a 3D scene of your own place.

EyeForge trains vision models for low-power hardware like the ESP32-CAM or a phone, starting from a plain description and a handful of photos instead of thousands of labeled images.

An interview with NVIDIA Nemotron 3.5 Super VL (Nebius Token Factory) looks at a sample of the user's photos and turns their needs into a dataset spec: classes, place, camera and target hardware, with real object sizes looked up on the web with Tavily. A teacher model (SAM 3 or SigLIP 2) labels the user's raw photos, learning their criterion from a few corrections. One of the photos becomes a walkable 3D scene of the user's own place (WorldStereo, MapAnything, gsplat, NVIDIA Fixer), and NVIDIA Isaac Sim renders the objects to learn inside it, on Nebius AI Cloud GPUs. The result is a quantized int8 model small enough to run on a microcontroller, flashed from the browser.

In our first experiment (vehicle classes on real traffic-camera images), 50 real photos per class alone reached 58% accuracy; adding our synthetic data raised it to 77%, with the same photos. Synthetic data completes the user's photos, it does not replace them.
