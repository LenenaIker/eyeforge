# Prueba de viabilidad: notas

Objetivo: medir si un clasificador tiny entrenado solo con datos sintéticos (Isaac Sim) funciona con imágenes reales de tráfico.

## Pipeline

1. `download_assets.py`: modelos 3D de Objaverse por categoría LVIS (solo licencias CC-BY, CC-BY-SA, CC0).
2. `convert_assets.py` (Isaac Sim): GLB a USD, escala real por clase.
3. `generate.py` (Isaac Sim + Replicator): carretera con arcén, cámara de poste 2-6 m, cielos HDRI, árboles y personas como distractores. Modo `preview` para revisar assets uno a uno.
4. `prepare_crops.py`: recortes 96x96 de la zona de interés (sintético) y conjunto real MIO-TCD.
5. `train.py`: MobileNetV2 alpha 0.35, 96x96, int8 (TFLite Micro). Se evalúa el modelo int8 sobre reales.

## Hallazgos

- **El catálogo público de NVIDIA no tiene vehículos civiles modernos** (solo una carretilla y material militar/WWII). Objaverse-LVIS sí: 1156 categorías, 30-120 modelos por tipo de vehículo. Sirve también como catálogo del producto.
- **~30% de los modelos de Objaverse no valen**: peanas y dioramas, modelos tumbados o boca abajo, categorías LVIS mal asignadas, objetos que no son vehículos. Se detectan a simple vista en el preview: es exactamente el paso de control de calidad que haría Nemotron VL en el producto.
- **Revisión humana de assets (2026-09-23)**: el usuario revisó los 270 con la página "Criba de vehículos". Coincidencia con Claude: 239/270 (88.5%). Claude fue demasiado estricto en 20 (sobre todo buses, caravanas y grúas en vehículo grande, y motos estilizadas) y demasiado permisivo en 11 (mayoría en vehículo grande: modelos que en realidad son furgonetas o pickups mal etiquetadas por LVIS, y motos de ciencia ficción). La clase "vehículo grande" es la más ambigua: ahí está el 61% de los desacuerdos. Tras la revisión quedan 193 válidos (63 pequeños, 42 grandes, 49 motos, 39 bicis).
- **Escalado (arreglado 2026-09-24, `convert_assets.py` v2)**. Cuatro problemas encadenados:
  1. Modelos girados dentro del archivo: el bbox alineado a ejes se inflaba. Ahora rectángulo de área mínima en planta (envolvente convexa + calibres rotatorios) y se endereza el modelo con el eje largo en X.
  2. El conversor GLB->USD mete su propia rotación en el prim raíz y no conmuta con la nuestra. El marco "de pie" es `p_local * to_z * R_raiz`; el envoltorio anula los xformOps de la raíz y aplica una sola matriz. Esto también explicaba el desplazamiento que había que corregir en tiempo de ejecución.
  3. Algunos modelos traen planos invisibles (suelo o recogedor de sombras) mayores que el vehículo: se ignoran planos grandes sin grosor al medir.
  4. Vehículos de 4 ruedas se escalan por el ANCHO (normalizado: ~2.5 m pesados, 1.7-1.9 turismos), no por el largo (un bus articulado mide 18 m). Motos y bicis por el largo.
  Verificación: 266/270 modelos cuadran con los vértices compuestos exactos (el bbox de USD de objetos girados es conservador y NO sirve para verificar). Aceptados fuera de proporciones reales: 42 -> 18 (formas raras de verdad: coche de 1910, tuk-tuk, camiones de dibujos).
- **Modelos sin color**: 23 de 270 no traen color en origen (no es fallo de conversión: las 162 texturas llegan bien a USD). Aleatorización de pintura en `generate.py`: siempre a los sin color, 35% al resto, paleta de pinturas reales, material ligado con strongerThanDescendants.
- **Reclasificación**: SUV "SCSO Suburban" y pickup "Carro" pasan de grande a pequeño (`data/usd/class_override.json`).
- **La escala precalculada no sobrevive a la composición USD** en algunos modelos: se mide y se corrige el bbox en tiempo de ejecución dentro de Isaac.
- **Assets remotos que aún no han cargado miden 0**: con escala 1 un árbol en cm mide kilómetros. Hay que esperar a la carga y descartar lo que no llegue.
- **Los HDRI remotos se cargan de forma asíncrona**: la captura sale negra o blanca. Solución: HDRI en local, carga síncrona de texturas y cambiar el cielo cada N frames.
- **Diferencias clave con el mundo real (MIO-TCD)**: motos y bicis reales siempre llevan persona encima; cámaras de tráfico más altas y picadas; desenfoque, JPEG y colores lavados.

## Resultados (test real MIO-TCD, 400 por clase, modelo int8)

| Entrenamiento | Acc. real int8 | Acc. real float |
|---|---|---|
| Solo 50 reales por clase (referencia) | 0.584 | 0.655 |
| Solo sintético (v1) | 0.500 | 0.595 |
| Solo sintético (v2, resolución igualada) | 0.539 | 0.533 |
| Sintético + 50 reales por clase (v1) | **0.794** | 0.793 |
| Sintético + 50 reales por clase (v2) | 0.757 | 0.771 |

Azar: 0.20.

### v3 (2026-09-24): assets revisados, escalado arreglado, pintura aleatoria. 3 semillas por configuración

| Entrenamiento | Acc. real int8 (media ± desv., 3 semillas) |
|---|---|
| Solo sintético v2 | 0.543 ± 0.037 |
| Solo sintético v3 | 0.545 ± 0.034 |
| Híbrido v2 | 0.768 ± 0.010 |
| Híbrido v3 | 0.773 ± 0.017 |

**Sin diferencia medible entre v2 y v3.** El ruido entre semillas en solo sintético es de unos ±3.5 puntos: las comparaciones de una sola ejecución de la tabla anterior no eran fiables. Las mejoras de calidad de assets no son el cuello de botella en MIO-TCD.

**El error dominante es la bici con ciclista**: en solo sintético v3, de 400 bicis reales ~206 se clasifican como moto y solo ~141 como bici. Un ciclista montado se parece más a una moto que a una bici vacía. Siguiente palanca clara: personas montadas.

### Lectura

- **Solo sintético todavía no basta** (54% frente al 58% de 50 fotos reales).
- **Híbrido funciona muy bien**: los sintéticos suman unos 20 puntos sobre las mismas 50 fotos reales. El producto debe venderse como "pocas fotos reales + sintético", no como "cero fotos".
- v1 aprendió un atajo: "borroso = dos ruedas" (en el render motos y bicis salen pequeñas). v2 lo corrige igualando la resolución efectiva; la confusión restante es coche frente a camión.

### Siguientes palancas (por impacto esperado)

1. Personas montadas en motos y bicis (en el mundo real siempre las llevan).
2. Cámaras más altas y picadas (MIO-TCD son cámaras de tráfico de 6-15 m).
3. Texturas reales de asfalto y arcén en vez de colores planos.
4. Evaluar con fotos de la propia ESP32-CAM: MIO-TCD es solo un proxy.

## Paso al paquete `sim2chip` (2026-09-24)

Mismo experimento híbrido lanzado con `sim2chip run specs/lora_alarm_traffic.json` de principio a fin, sin pasos a mano (job en `data/jobs/lora_alarm_traffic/`):

| | Acc. real int8 (3 semillas) | Por semilla |
|---|---|---|
| Híbrido v3 (spike) | 0.773 ± 0.017 | 0.785 / 0.754 / 0.781 |
| Híbrido sim2chip | 0.764 ± 0.001 | 0.765 / 0.764 / 0.763 |

Dentro del ruido: criterio de la semana 1 cumplido. Diferencias con el spike: distancia de cámara deducida del tamaño real y del FOV (ya no por clase), escala por asset con otra semilla, reparto test/train de MIO-TCD distinto (otro orden aleatorio). Los mismos 274 modelos seleccionados y los mismos 193 tras la revisión. La precisión por clase varía mucho entre semillas en ambos (bicis: 59-86% en el spike, 81-90% ahora): no sacar conclusiones por clase con 3 semillas.

Tiempos en la 3090: fetch 3 min (con caché), convert 5.5 min (solo normalización, crudos en caché), render 16 min (8000 frames a 320x240), entrenamiento de 3 semillas unos 15 min.

## Semana 2: QA automático de assets con Token Factory (2026-09-24)

En Token Factory ningún Nemotron acepta imágenes (Nano 30B, 3.5 Lightning, Super 120B y Ultra 550B son solo texto). Modelos con visión: MiniCPM-V-4.5, GLM-5.3-Flash, DeepSeek-V4.1-Flash, Kimi K2.6 y K3. Medido contra la revisión humana de los 270 modelos de tráfico (`scripts/qa_agreement.py`, veredictos en `data/jobs/lora_alarm_traffic_autoqa/qa_runs/`), una ejecución por opción, temperatura 0:

| Estrategia | Modelos | Acuerdo | Descartes humanos detectados (de 77) | Aceptados humanos descartados (de 193) | Tiempo | Coste |
|---|---|---|---|---|---|---|
| vision | MiniCPM-V-4.5 | 76.7% | 16 | 2 | 36 s | ~0.15 $ |
| vision | GLM-5.3-Flash | 88.9% | 56 | 9 | 16.5 min | ~0.15 $ |
| describe_then_decide | MiniCPM-V describe, Nemotron Super decide | 84.4% | 54 | 19 | ~2 min | ~0.22 $ |
| Referencia | Claude (revisión del spike) | 88.5% | | | | |

- Nemotron como juez sube MiniCPM de 76.7% a 84.4%: MiniCPM ve bien pero decide mal (se contradice: "tiene peana" y aun así "keep").
- GLM razona mucho por imagen (unos 540 tokens de salida y 21 s por llamada con 8 en paralelo).
- Diferencias de 2 a 4 puntos (5 a 10 modelos) pueden ser ruido con una sola ejecución.
- GLM-5.3-Flash describe + Nemotron Super decide: 83.3% (55 detectados, 23 aceptados descartados, ~6 min, ~0.22 $). Quién describe apenas cambia el resultado: el cuello de botella es la decisión sobre texto.
- Afinado del criterio de Nemotron sobre las descripciones guardadas de MiniCPM (`scripts/qa_redecide.py`, ~0.12 $ y 1.5 min por intento), reglas generales, no de vehículos: intento 1 (objeto completo frente a fragmento, variantes cercanas a la clase más parecida, diseños raros pero posibles) 86.7%; intento 2 (las marcas del descriptor mandan, ficha y descripción que no cuadran = descartar) 87.4%.
- Decisión (usuario, acordada antes de medir): si el afinado no llegaba al nivel de GLM solo, estrategia por defecto "vision" con GLM-5.3-Flash. Se quedó a 1.5 puntos (4 modelos), así que el defecto es GLM solo; `describe_then_decide` (MiniCPM + Nemotron, unas 8 veces más rápido) queda como opción.
- Gasto total en Token Factory del día hasta aquí: ~0.51 $ más ~0.5 $ de las pruebas posteriores.

## Semana 2: profesor (etiquetado de fotos crudas) con GLM-5.3-Flash (2026-09-24)

200 fotos de MIO-TCD no usadas en el trabajo de tráfico (40 por clase, carpeta plana sin la clase en el nombre), `scripts/label_agreement.py`:
- Aparta como dudosas 108 (54%): 75 con "unsure" explícito ("muy borrosa y pixelada") y 33 por confianza < 0.6 (en esas acertaría solo 20/33).
- Acierta 89 de las 92 que etiqueta: **96.7% de precisión**. Etiquetas útiles correctas: 89/200.
- Las dudosas son las diminutas: lado menor mediano 62 px frente a 94 px de las etiquetadas. MIO-TCD son recortes pequeños de cámaras de tráfico; con fotogramas completos de la ESP32-CAM se esperan muchas menos dudas.
- Umbral 0.6 se mantiene: prima la precisión sobre la cantidad. 3 min 45 s, ~0.03 $.

## Imagen a 3D con TRELLIS v1 (2026-09-25)

Objetivo: generar assets 3D a partir de fotos cuando la búsqueda en internet no encuentra el objeto. Script `spike/img2mesh/run_trellis.py` (entorno conda `trellis2`; código, extensiones y pesos en `/media/lenena-iker/Crucial/tools/`, cachés con `HF_HUB_CACHE` y `TORCH_HOME` apuntando allí). Resultados y hojas en `data/img2mesh_test/out_trellis1/`.

- Flujo: OWLv2 detecta el objeto por texto y lo recorta, BiRefNet quita el fondo y TRELLIS genera la malla con textura; se exporta un GLB ligero (textura 1024, 2k-16k caras, ~1 MB).
- Descartados: TRELLIS.2 (necesita DINOv3, gated en Hugging Face; el acceso no llegó) y el quitafondos por defecto de TRELLIS.2, RMBG-2.0 (gated y CC BY-NC). Hunyuan3D: su licencia excluye la UE. El código y los pesos usados son MIT o Apache, PERO la exportación a GLB (`to_glb`) usa nvdiffrast y el rasterizador de mip-splatting, ambos solo no comerciales (corregido 2026-09-26, ver TODO 3.E).
- Coste en la 3090: unos 7-9 s de generación + 8-11 s de exportación por modelo, ~10 GB de VRAM.
- 10 fotos de Wikimedia Commons (licencias en `data/img2mesh_test/inputs/SOURCES.txt`): llaves inglesas 3/3 muy buenas (fotos de producto, fondo liso); cámaras de seguridad 2 buenas, 1 con flecos largos en el soporte (objeto diminuto en la foto) y 1 fallida (contrapicado, sale tumbada sobre una lámina); abejas 1 fallida (láminas planas aunque el recorte era limpio), 2 aceptables (alas pobres).
- Lectura: objetos rígidos y compactos salen muy bien; lo fino u orgánico y los objetos pequeños en la foto, peor. Los fallos son evidentes a la vista: los puede filtrar la QA con visión existente, y repetir con otra semilla cuesta ~17 s.

## Prueba del usuario: macetas desde la web (2026-09-26)

Primera prueba de principio a fin como usuario real: entrevista en la web, 37 fotos del móvil sin ordenar (terrazas, interiores, fachadas y calles; 6 de ellas para sacar 3D), perfil `generic`. Trabajo `data/jobs/maceta_counter`. Resultado: 64.7 ± 4.8% en 17 fotos reales de test (semillas 70.6 / 58.8 / 64.7), validación sintética 100%.

Hallazgos, por impacto:
1. Tarea mal planteada para este caso: el alumno entrena con recortes centrados en la maceta (`roi_classification`), pero las fotos reales entran enteras y en muchas la maceta es pequeña. Falla justo las macetas pequeñas en escenas amplias (garaje, muro). Para "¿hay una maceta en la foto?" hace falta otra tarea: imagen entera, por trozos o detector.
2. La clase de fondo sintética es trivial: los recortes "empty" son trozos de suelo liso, sin nada. En la realidad las fotos sin maceta están llenas de cosas (fregona y cubos, árboles, fachadas): el modelo aprende "hay un objeto = maceta" y da falsos positivos. Por eso la validación da 100% y no sirve para elegir modelo.
3. Etiquetas del profesor ambiguas: su prompt pide la clase del objeto PRINCIPAL, así que las fachadas con macetas pequeñas salen "empty" casi siempre, pero no siempre (etiquetado inconsistente en el límite). La definición de la clase tiene que venir del usuario (¿cuenta una maceta pequeña al fondo?) y la entrevista debe preguntarlo.
4. Fuga leve: 2 de las 6 fotos usadas para generar el 3D acabaron en test (ambas aciertan). Las fotos de `assets.photos` no deberían ir a test.
5. Render: props con escala mal (una "barandilla" de madera gigante), sillas de oficina como props de terraza, macetas generadas algo oscuras en los recortes.

Lo que sí funcionó: la entrevista corregida (fotos crudas al profesor, consulta en inglés, suelo de baldosa), el profesor (21 maceta / 14 empty / 2 dudosas) y los 6 modelos 3D desde las fotos del usuario.

## Tarea de conteo: primera prueba con las macetas (2026-09-26)

Nueva tarea `task: "count"` (contar y localizar, estilo FOMO): MobileNetV2 cortada a paso 8 (`block_6_expand_relu`) y cabeza 1x1, salida en rejilla (28x28 a 224 px, 12x12 a 96 px), 133 KB en `generic`, 58 KB en `esp32cam`. En datos de juguete (cuadrados rojos) F1 0.98-1.0 en int8, así que el modelo, la cuantización y la decodificación funcionan.

- Profesor: GLM-5.3-Flash da cajas en PÍXELES de la imagen que recibe aunque se le pidan coordenadas 0-1000 (con 0-1000 salen desplazadas). Con píxeles y la imagen a 1024 px marca muy bien, incluso macetas de fachada lejanas: 213 macetas en 37 fotos, 0 fallos. Errores típicos: alguna caja en fachadas donde no se distingue nada y plantas grandes encuadradas enteras. Tarda 5-40 s por foto.
- Render: 0-6 macetas por frame repartidas alrededor del punto al que mira la cámara, cajas por instancia con `occlusionRatio` de Replicator. Funciona.
- Peso de las celdas con objeto: con 100 (el de FOMO) el modelo dispara en todas partes (19 macetas por foto para 5 reales, incluso en el cielo). Con 20 el error de conteo baja de 14.9 a 3.7 por foto; con 5 apenas detecta. El umbral se elige ahora en validación (sale 0.6).
- Resultado real (18 fotos de test): F1 de puntos 0.25 ± 0.03, error de conteo 3.7 por foto; validación (sintético + reales repetidos) F1 0.59-0.63. El límite ya no es el modelo sino el dominio y la resolución: el sintético no tiene fachadas ni balcones y a 224 px una maceta de fachada ocupa 2-5 px. Siguiente: escenas desde las fotos del usuario (fachada procedural) y más resolución o inferencia por trozos en el móvil.

## Nebius: arranque de 4 jobs en paralelo (2026-09-26)

`scripts/nebius_parallel_probe.py --jobs 4` (imagen de Isaac Sim, L40S en eu-north1, on-demand). Resultado: los 4 estuvieron ~25 min en PROVISIONING; 3 acabaron en ERROR "NotEnoughResources: VM schedule timeout, most likely due to insufficient hardware resources" y 1 arrancó (imagen descargada en ~2.5 min). La cuota (32 L40S) no garantiza capacidad: en eu-north1 no hay L40S on-demand libres de forma fiable. Repartir el render en muchas GPUs no es viable así; hay que mirar reservas, otras regiones o plataformas, o un worker caliente que se quede con una GPU una vez conseguida.

## Escenas desde las fotos: fachadas, balcones, interiores (2026-09-26)

- Perfil de escena (`sim2chip describe-scene`, acción `describe_scene` de la entrevista): GLM describe 16 fotos al azar (contexto, distancia, apoyos, pared, barandilla) y se agregan en `scene.contexts`, `wall_tags` y `railings`. En las macetas: 70% fachada (3-23 m; barandilla, suelo de balcón, repisa, acera), 20% balcón de cerca, 10% interior; revoque, ladrillo, piedra; barrotes y obra. Cuadra con las fotos.
- Generador procedural (`scenes/places_layout.py`, Python puro y testeado; `scenes/places.py` en Isaac): edificio de 3-6 plantas y 3-6 vanos con balcones (barrotes u obra, a veces toldo), ventanas con repisa, puertas y acera; habitación sin techo con ventana, repisa y muebles. Una malla por material reescrita cada 20 frames; texturas de pared de Poly Haven. Los balcones e interiores salían casi negros: una luz de relleno junto a la cámara lo arregla.
- Resultado (mismo spec, 2000 frames, 3 semillas; el test no es idéntico porque GLM volvió a marcar): F1 de puntos 0.25 -> 0.35 ± 0.03, error de conteo 3.7 -> 3.1 por foto. Mejoran las fachadas; fallan las macetas grandes de cerca (0 encontradas en 3 fotos): el objetivo es una sola celda y un objeto que llena media imagen no tiene centro claro para el modelo. Siguiente: marcar el núcleo de la caja (ya hay cajas en sintético y del profesor).

## Test fijo (2026-09-27)

Las respuestas del profesor van a una caché compartida (`cache/labels/<prompt>/<modelo>/<foto>.json`, por contenido de la foto) y ganan a las antiguas de cada job; el reparto train/test es por huella del contenido; las fotos de las que salió un 3D van siempre a train; aviso con menos de 20 fotos de test. Comparación limpia sobre las mismas 18 fotos y 103 macetas (3 semillas cada uno):

| job | F1 de puntos | error de conteo por foto | conteo exacto |
|---|---|---|---|
| `macetas_count` (solo suelo) | 0.234 ± 0.029 | 3.85 | 7% |
| `macetas_places` (fachada, balcón, interior) | 0.336 ± 0.008 | 4.28 | 11% |

Las escenas desde las fotos suben el F1 10 puntos y lo hacen más estable entre semillas, pero el error de conteo no mejora (algo peor): acierta más macetas en su sitio y a la vez se equivoca en el número total en las fotos con muchas. 18 fotos siguen siendo pocas: una foto mueve el F1 unos 5 puntos.

## Macetas grandes de cerca: marcar el núcleo de la caja (2026-09-27, resultado negativo)

Idea: entrenar marcando el núcleo de la caja (40% de ancho y alto) en vez de la celda central, para que las macetas grandes se aprendan como región. Mismo test (18 fotos, 103 macetas), 3 semillas, métrica nueva (acierto dentro del núcleo, más justa con los objetos grandes):

| entrenamiento | F1 | error de conteo | 5 fotos cercanas: F1 | error |
|---|---|---|---|---|
| celda central | 0.345 | 4.3 | 0.32 | 1.8 |
| núcleo 40% | 0.304 | 4.9 | 0.37 | 4.3 |

El núcleo de una maceta grande se rompe en varias manchas y cada una cuenta como un objeto: más aciertos de posición, peor conteo. Queda desactivado (`TRAIN_CORE = 0`); se conservan el tamaño de caja en los puntos y la métrica con núcleo. Reentrenado: F1 0.341 ± 0.017, error 4.3. El error de conteo grande viene sobre todo de las fachadas con 10-14 macetas donde encuentra 3-4: a 224 px son pocos píxeles; la palanca siguiente es la resolución (inferencia por trozos).

## Nebius: todo el pipeline en un solo job (2026-09-27)

`--backend nebius` lanza ahora un único job (modo worker, `cloud_worker.py`) con la imagen oficial de Isaac Sim; al arrancar instala con `uv` un Python 3.11 con TensorFlow y ejecuta el pipeline entero con el backend local. Sube progreso (log y marcadores) cada 20 s; al final baja el modelo, los JSON y lo que usa la review (48 frames de muestra). El modo antiguo (un job por etapa) queda con `SIM2CHIP_NEBIUS_MODE=stage`.

Prueba con `specs/macetas_cloud.json` (macetas con fachadas, 2000 frames, 3 semillas): COMPLETED en 23.8 min, unos 0.61 $, F1 0.338 ± 0.021 (igual que en local). Antes: 55 min con un job por etapa; en local unos 40 min. Conseguir la GPU, bajar la imagen e instalar el entorno llevó menos de 6 min (un intento fallido por un bug costó 0.14 $). Fallos corregidos por el camino: la subida no entraba en carpetas enlazadas (`--input`), el log de la ejecución anterior se mostraba como progreso y el marcador de render no volvía.

## Telemetría: a dónde van los 26 min de una ejecución en Nebius (2026-09-27)

`logs/timeline.jsonl` guarda ahora la duración de cada etapa, de las fases del worker y de cada estado del job de Nebius; al final se imprime un desglose. Ejecución completa (`--force`) de `specs/macetas_cloud.json`: 25.8 min, 0.65 $, F1 0.344.

- train 8.4 min (3 semillas en serie)
- convert_assets 4.9 min: se repite entera cada vez porque el worker solo baja `db/` y `labels/` de la caché, no las conversiones USD ya hechas que están en el bucket
- esperar GPU + arrancar + bajar imagen 4.0 min (1.1 + 0.7 + 2.2)
- render 3.6 min (2000 frames a 640x480 en L40S)
- instalar el entorno con uv 1.6 min; bajar datos del job 0.8 min (10.200 ficheros, incluidos resultados viejos que `--force` va a rehacer)
- points 0.9 min; el resto (profesor con caché, ingest, QA, evaluate, export) menos de 1 min

Arreglos baratos, unos 26 -> 13 min: bajar las conversiones USD de los assets del job, semillas en paralelo en la misma GPU (o una sola en modo rápido), no bajar lo que se va a rehacer. Con worker ya encendido e imagen propia con el entorno: unos 7-8 min.
