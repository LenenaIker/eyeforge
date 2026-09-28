# Modelos candidatos para el profesor (etiquetado)

Investigación del 2026-09-28. Fuente: respuesta de GPT al prompt de comparación (sin verificar todavía) y prueba propia con 4 fotos de macetas en Token Factory.

## Problema

El profesor actual (GLM-5.3-Flash, VLM de chat en Token Factory) encuentra la mayoría de los objetos, pero sus cajas son flojas y a veces encaja la planta en lugar de la maceta. Un VLM de chat genera coordenadas como texto: no es un buen generador de propuestas cuando hay 20 objetos pequeños por foto. Parte del fallo era nuestro: `COUNT_PROMPT` pedía "a pot and the plant growing in it".

## Prueba en Token Factory (2026-09-28)

Solo 4 modelos aceptan imágenes: GLM-5.3-Flash (el mejor, ~7 s/foto), Kimi-K3 (algo más fino, ~30 s y falla la mitad), Gemma 3 27B (cajas en sitios vacíos) y MiniCPM-V 4.5 (respuestas rotas). Qwen3.5, Qwen3.8 y GLM-5.3 no aceptan imágenes allí.

## Candidatos (según GPT, por verificar)

| Modelo | Tipo | Licencia | Por qué interesa |
|---|---|---|---|
| **SAM 3** (Meta, nov 2025) | Segmentación por concepto (texto o ejemplar) | SAM License (propia, revisar uso comercial) | Diseñado para devolver *todas* las instancias de un concepto; admite ejemplar positivo (una caja de maceta) |
| **LocateAnything-3B** (NVIDIA, may 2026) | VLM de detección densa | NVIDIA, no comercial | Entrenado para detección densa y etiquetado de datasets; es de NVIDIA (encaja con el hackathon) |
| **Qwen3-VL-8B** | VLM con localización | Apache 2.0 | RefCOCO ~89.7; buen verificador ("¿es la maceta y no la planta?") |
| **Rex-Omni-3B** (CVPR 2026) | Híbrido detector/VLM | revisar | Comparable a Grounding DINO en COCO/LVIS |
| **Grounding DINO-B** + teselas | Detector texto a cajas | Apache 2.0 | Línea base abierta y reproducible |
| Gemma 4 E4B (mar 2026) | VLM general | Apache 2.0 | Dice soportar detección y señalado, sin números de localización |

## Diseño recomendado

Detector que maximiza la recuperación (SAM 3 o LocateAnything) y VLM que verifica cada candidato con la descripción del usuario y las reglas de la calibración. Teselado (2x2) para objetos de 15 a 30 px. Correr en GPU propia de Nebius AI Cloud (ninguno está en Token Factory): coste estimado por GPT de 0,10 a 0,35 $ por 100 fotos en H100, dominado por el arranque.

## Pendiente

- Verificar existencia, licencias y precios de Nebius (L40S y H100 por hora).
- Benchmark propio: fotos de macetas con las correcciones de la calibración como verdad; medir recuperación por instancia y confusión planta/maceta.
