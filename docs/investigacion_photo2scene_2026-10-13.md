# Investigación: estado del arte para photo2scene (2026-10-13)

Contexto: la ruta de calle se encendió en producción (c5acbee). Prueba con fachadas de macetas (`p2s_test/compare/maceta_calle.html`): la fase 2 (imitación) gusta al usuario; fallan los huecos de ventana, la textura que no encaja, la extracción de objetos en balcones y la fase 3 (cruce). Aquí, qué hacía Mira-Scene, qué hacemos nosotros y qué se publica ahora.

## Qué hacía Mira-Scene y qué nos quedamos

[Mira-Scene](https://arxiv.org/pdf/2609.23796) (VAST, MIT): SAM 3 + VLM (objetos y apoyos) + MoGe-2 + CCM (mapa de coordenadas canónicas: por píxel, qué punto de la malla canónica se ve) + SAM 3D + suelo plano + grafo de apoyos + panorama de entorno generado. La pose sale de un RANSAC de similitud entre el CCM y la nube de puntos, por eso aguanta objetos cortados o tapados.

Lo nuestro (NOTES, 2026-10-05 a 10-08): nos quedamos con la idea (máscaras, profundidad, malla por instancia, colocación por geometría) y la sustituimos pieza a pieza: SAM 3D completo con nuestra profundidad como point map, `flow_refine` (ICP con escala sobre los puntos de MoGe), suelo y fachadas planas por RANSAC con material de la foto, casas como mallas SAM 3D con la foto proyectada por vértice (`env_add --photo-texture`), sol por geometría de sombras, fotometría por proporciones, borrado con Qwen. Mira gana en interiores con muebles (CCM); en calle no monta edificios (sin suelo visible no hace escena). Sus etapas de entorno usan APIs de pago.

## Lo publicado más cercano (2025-2026)

| Trabajo | Qué hace | Útil para nosotros | Código y licencia |
|---|---|---|---|
| [Building Rome from a Single Image](https://arxiv.org/html/2610.08790v1) (oct 2026) | TRELLIS 2 con LoRA, condicionado por el point map de MoGe-3 y DINOv3 por vóxel (solo cerca de la superficie vista, con residuo de profundidad firmado); la escena en trozos adaptativos (más grandes lejos) generados en autoregresión con solape | Geometría de escena entera exterior de una foto, edificios incluidos. Supera a Lyra 2.0, 3D-RE-GEN y otros en Tanks and Temples (Chamfer -24 %) | No dice; sin textura (solo geometría) |
| [SimuScene](https://arxiv.org/html/2606.03994v1) (jun 2026) | Editor MLLM borra objetos, SAM 3D estructuras fijas, MoGe-2, FoundationPose, MuJoCo como sonda: si un objeto cae, se estira o se regenera | Comprobación física de "flotar" y "enterrado" barata y objetiva; anclajes de objetos colgados por clasificación VLM (de pie o colgado) | Página del proyecto; CC BY |
| [Lucida](https://arxiv.org/abs/2608.30821) (ago 2026) | Parse, generate, place desde vídeo; el VLM coloca moviendo el gizmo en bucle cerrado | Colocación con VLM en bucle cerrado; solo interior | Sin código confirmado |
| [SVI2LoD3](https://arxiv.org/abs/2608.29992) (2026) | Ventanas y puertas de fachada desde street view con segmentación zero-shot y VLM, a modelos LoD3 | Detectar huecos de fachada como piezas con jerarquía | [GitHub](https://github.com/hcu-cml/citydb-SVI2LoD3-ai) |
| [DiffTex](https://arxiv.org/abs/2509.23336) (TOG 2025) | Textura por téxel como mezcla ponderada de píxeles de fotos; los pesos se optimizan con render diferenciable (consistencia fotométrica y de perspectiva, sin costuras) | Justo nuestro caso: un proxy simplificado de edificio con fotos encima | Sin confirmar |
| [GOATex](https://arxiv.org/html/2511.23051) (2025) | Capas de visibilidad por rayos, textura por capa, mezcla en UV | Lo oculto se rellena por capas, no con un color medio | Sin confirmar |
| MVPaint (CVPR 2025) | Rellena lo no visto en 3D, propagando color por distancia y normal | Relleno de caras no vistas más coherente que el tono medio | Sin confirmar |
| [ACDC, primos digitales](https://arxiv.org/abs/2410.07408) (CoRL 2024) | Varias escenas parecidas en vez de un gemelo exacto | Respalda la extrapolación (fase 3) como variación, no copia | Abierto |
| [Gaussian splatting como generador de datos para detección](https://arxiv.org/pdf/2504.16740) | Insertar objetos en escenas reconstruidas con 3DGS para entrenar detectores | Alternativa a la malla para la fase 2, sin mallas que encajar | Sin confirmar |

Modelos de mundo desde una foto: HY-World 2.0 (licencia que excluye la UE), Lyra 2.0 y SHARP (no comerciales), Marble (cerrado). TRELLIS.2 y TripoSplat son MIT, pero de objetos, no de escenas.

## Las matemáticas que se usan

1. **Geometría monocular métrica**: point maps afines más escala (MoGe-2/3), intrínsecos de la red. Error típico en exterior de 5-10 %: por eso todos los métodos refinan después contra la nube de puntos.
2. **Registro malla-foto**: RANSAC de similitud con correspondencias densas (CCM de Mira), ICP con escala (nuestro `flow_refine`), o pose por FoundationPose. La textura descolocada que vemos nace aquí: si la malla no cae sobre su píxel, la proyección pega la ventana en la pared.
3. **Fachadas**: mundo Manhattan, puntos de fuga, homografía a vista frontal (ya rectificamos superficies en `piece_build`). Encima, parsing de fachada (rejilla de ventanas) y gramáticas de forma (Müller 2007; hoy con VLM y segmentación zero-shot, SVI2LoD3). Un hueco de ventana es una extrusión hacia dentro de 10-30 cm: con la máscara de ventana y el plano de la fachada se construye sin aprender nada.
4. **Texturizado proyectivo**: visibilidad por z-buffer, selección de vista por cara (MRF, Waechter 2014), nivelación de color entre vistas, y hoy optimización con render diferenciable (DiffTex: pesos por téxel). Lo no visto se rellena en 3D (MVPaint) o por capas (GOATex), no con un tono medio.
5. **Física como control**: soltar con gravedad y medir el desplazamiento normalizado (SimuScene, ρ < 0,15 se estira, si no se regenera). Es una comprobación simbólica, más fiable que un VLM (`docs/escena_agente_literatura.md`).
6. **Generación condicionada por geometría**: difusión o flow matching 3D con la nube de puntos como condición por vóxel, solo cerca de la superficie vista (Building Rome). Es la evolución de "SAM 3D por instancia" hacia "la escena entera de una vez".

## Qué encaja con nuestros fallos

- **Huecos de ventana y puertas**: la fachada vista no viene de SAM 3D sino de la propia profundidad. Plano de fachada por RANSAC, máscaras de SAM 3 para "window" y "door" (o "balcony"), y extrusión hacia dentro con el fondo medido por MoGe dentro de la máscara (o un valor fijo de 15-25 cm si MoGe es plano). Barato, determinista, sin modelos nuevos. SAM 3D se queda para lo no visto (traseras, tejado).
- **Textura que no encaja**: medir la desalineación (silueta y bordes de la malla proyectada frente a la foto) y corregirla antes de proyectar. Opciones de menos a más: (a) proyectar solo en caras cuya profundidad renderizada coincide con MoGe (ya filtramos por máscara erosionada, falta por profundidad); (b) usar como superficie vista una malla 2,5D sacada directamente de la profundidad de MoGe (la textura cae siempre en su sitio porque la geometría sale del mismo píxel), unida a SAM 3D por detrás; (c) optimizar con render diferenciable un pequeño desplazamiento de UV o de pose por pieza (estilo DiffTex).
- **Macetas en balcones**: hoy `kept_inside` las deja dentro de la casa. Tratar "lo que está sobre un balcón o alféizar" como objetos con soporte propio (el balcón es un soporte horizontal más, como la mesa en la ruta de soporte), y SimuScene da la idea de anclar lo colgado.
- **Fase 3 (cruce)**: casas flotando y asfalto vacío. La comprobación física (soltar y medir) atraparía lo que flota; los primos digitales (ACDC) dicen que variar es mejor que copiar exacto.
