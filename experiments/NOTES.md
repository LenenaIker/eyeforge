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

Mismo experimento híbrido lanzado con `sim2chip run specs/examples/lora_alarm_traffic.json` de principio a fin, sin pasos a mano (job en `data/jobs/lora_alarm_traffic/`):

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

Objetivo: generar assets 3D a partir de fotos cuando la búsqueda en internet no encuentra el objeto. Script `experiments/img2mesh/run_trellis.py` (entorno conda `trellis2`; código, extensiones y pesos en `/media/lenena-iker/Crucial/tools/`, cachés con `HF_HUB_CACHE` y `TORCH_HOME` apuntando allí). Resultados y hojas en `data/img2mesh_test/out_trellis1/`.

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

`experiments/nebius_parallel_probe.py --jobs 4` (imagen de Isaac Sim, L40S en eu-north1, on-demand). Resultado: los 4 estuvieron ~25 min en PROVISIONING; 3 acabaron en ERROR "NotEnoughResources: VM schedule timeout, most likely due to insufficient hardware resources" y 1 arrancó (imagen descargada en ~2.5 min). La cuota (32 L40S) no garantiza capacidad: en eu-north1 no hay L40S on-demand libres de forma fiable. Repartir el render en muchas GPUs no es viable así; hay que mirar reservas, otras regiones o plataformas, o un worker caliente que se quede con una GPU una vez conseguida.

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

## Recorte de la ejecución en Nebius: primeras medidas (2026-09-27)

- `convert_assets` (4.9 min en la nube) no es lento por convertir: 190 s son el primer arranque de Isaac Sim en un contenedor nuevo (compila los shaders del renderizador; los arranques siguientes tardan 3 s) y 85 s la conversión de 41 modelos (unos 2 s cada uno). Traer del bucket los USD ya convertidos ahorra esos 85 s; el arranque en frío se pagaría igual en la siguiente etapa de Isaac. Por eso el worker guarda ahora las cachés de shaders (`~/.cache/ov/shaders`, `~/.nv/ComputeCache`, `~/.cache/nvidia/GLCache`) en el bucket de caché (`gpu/`) y las recupera al arrancar. Sin verificar en Nebius todavía.
- El worker planifica antes de bajar nada y solo trae lo que leen las etapas que van a correr (sus `after`, la entrada y lo que la etapa relee de sí misma, `rereads`, p. ej. los veredictos de `label` con las correcciones del usuario). Con `--force` en `macetas_cloud`: 38 ficheros en vez de 10.200.
- Semillas en paralelo (`SIM2CHIP_PARALLEL_SEEDS`, 3 por defecto), cada una en su proceso en la misma GPU. En la 3090 con 8 núcleos (`taskset`): 194 s en serie, 140 s en paralelo.
- El entrenamiento de conteo está limitado por CPU (aumento de datos imagen a imagen en `tf.data`): paso de 18 ms con 8 núcleos, 28 ms con 4, 72 ms con 2. En la L40S (8 vCPU) fue de 45 ms, así que allí las semillas en paralelo compiten por la misma CPU y ganarán menos. La palanca de verdad es hacer el aumento en la GPU por lote (recortes con `crop_and_resize`, color, ruido); el JPEG aleatorio solo existe en CPU. Cambia el entrenamiento: medir con 3 semillas antes de adoptarlo.

## Conversor GLB -> USD sin Isaac: `sim2chip/glb2usd.py` (2026-09-27)

Script del usuario con usd-core + trimesh, solo CPU y en varios procesos. Materiales glTF a UsdPreviewSurface, texturas en `<modelo>_textures/`, stage Z-up en metros. No soporta mallas con compresión Draco/meshopt ni animaciones.

- Velocidad: los 41 modelos de `macetas_cloud` en 1,4 s con 8 procesos (Isaac: ~85 s más el arranque de Kit). Los 536 GLB de la caché en 21 s con 16 procesos, 0 errores.
- Geometría: mismos triángulos y el mismo número de texturas en los 41. Isaac escribe en centímetros (`metersPerUnit` 0.01) y glb2usd en metros; no importa porque el render reescala cada modelo a su tamaño real del spec. Tras corregir la unidad, las medidas de `normalize()` coinciden en 38 de 41; las otras: una urna que Isaac deforma (sale tumbada; con glb2usd sale bien), una maceta (3%) y una silla (5%).
- Aspecto: `preview_assets` con los dos (mismo suelo, cielo, sol y vistas). Diferencia media por píxel menor de 3 en 35 de 41; las mayores son la urna de Isaac y giros distintos sobre el eje vertical (irrelevantes: el render gira los modelos al azar). Materiales, texturas y transparencias de hojas iguales a la vista.
- Efecto en el modelo (trabajos `cmp_glb2usd` y `cmp_isaac`, copias de `macetas_places`; mismo render, semillas y test de 18 fotos, en la 3090): F1 de puntos int8 0.348 ± 0.020 con glb2usd frente a 0.339 ± 0.023 con Isaac; error de conteo 4.6 frente a 4.2 por foto. Dentro del ruido: el conversor no cambia el resultado. Render 202 s frente a 204 s.
- Conclusión: glb2usd puede sustituir al Asset Converter de Isaac en `convert_assets`, que pasa a ser una etapa de CPU de segundos. Queda de respaldo Isaac para los GLB que trimesh no lea (Draco/meshopt).
- Integrado (2026-09-27): el script vive ahora en `sim2chip/glb2usd.py` (también `python -m sim2chip.glb2usd`). `convert_assets` es una etapa de CPU: glb2usd en paralelo a `usd/glb2usd/`, normalización también en paralelo (`_v3`), y el Asset Converter de Isaac (`sim2chip/isaac_convert.py`, en su propio proceso) solo para lo que glb2usd no lea y no esté ya convertido. Los 41 modelos desde una caché vacía: 3.6 s con 24 núcleos, 5.4 s con 8 (antes 296 s en Nebius).

## Tras la auditoría de código (2026-09-27)

Mismo spec (`macetas_places`), mismo test de 18 fotos, 3 semillas, en la 3090, tras sacar los valores de carretera del código (pintura por clase, distancia y apuntado de cámara según el objeto, nitidez por hardware): F1 de puntos int8 0.354 ± 0.013 (antes 0.348 ± 0.020), error de conteo 3.1 por foto (antes 4.6). El F1 queda igual dentro del ruido y el conteo mejora; probablemente porque la cámara ya no apunta por encima de las macetas y los fotogramas vacíos se hacen a la distancia de una maceta.

## Entrevista: ajustes de llamada de Nemotron 3 Super (2026-10-03)

Banco `experiments/interview_bench/`: la Session real contra 10 usuarios simulados (Qwen3-235B, en español, inglés, francés y alemán; ESP32-CAM, ESP32-S3, Jetson, PC), con el análisis de fotos simulado con respuestas fijas para que solo cambie Nemotron. Un juez (Qwen3-235B) marca cada pregunta como permitida, prohibida o repetida, y el spec final se compara con lo que dijo el usuario (tarea, hardware, fotos, orientación, `names`, `errors`, fondo). 20 entrevistas por variante (10 usuarios x 2), resultados en `runs/interview_bench/`.

| Variante | rechazados | formato | fallos spec | prohibidas | razonamiento/llamada | s | $ |
|---|---|---|---|---|---|---|---|
| A: razonamiento completo, T 0.3, máx. 4096 (antes) | 0.90 | 0 | 0.20 | 0.10 | 648 | 22.5 | 0.0082 |
| A8: A con máx. 8192 y llamadas cortas sin razonar | 0.40 | 0 | 0.15 | 0 | 621 | 17.9 | 0.0066 |
| B: `low_effort` | 1.05 | 0 | 0.45 | 0.25 | 98 | 12.7 | 0.0066 |
| C: sin razonamiento | 1.90 | 3.05 | 0.70 | 0.50 | 0 | 21.7 | 0.0305 |
| A8T: A8 con T 1.0, top_p 0.95 | 0.80 | 0.15 | 0.15 | 0.05 | 703 | 24.1 | 0.0085 |

- En Token Factory funcionan `chat_template_kwargs: {"enable_thinking": false}` y `{"low_effort": true}`; `reasoning_budget` se ignora (582 tokens de razonamiento con un límite de 256).
- Ninguna respuesta llegó a 4096 tokens: la diferencia A/A8 puede ser ruido.
- Adoptado A8: razonamiento completo y T 0.3 en la entrevista, `max_tokens` 8192 de margen, `english_queries` sin razonamiento. `low_effort` es casi el doble de rápido pero falla más en lo que decide solo (orientación, fondo, `errors`); T 1.0 rechaza el doble de specs.
- Sin razonamiento, una entrevista devolvió 59 veces el mismo JSON roto (60 turnos, 1.4 M tokens). Ahora la entrevista se corta tras 3 respuestas seguidas que no son JSON y le dice al modelo el error.
- Errores comunes a todas las variantes, para el prompt: falta `names` al contar (abejas, gallinas), `background_class` no es null en el césped, orientación "upright" con cámara cenital (galletas) y `errors` de la gatera.

## Seguir caminos: primera prueba con el dataset de caminos (2026-10-03)

`experiments/line_follow/follow.py`, sobre las 74 fotos de `data/real/roads` y las máscaras de SAM 3 de `road_grid`. Etiqueta sin humano: el camino principal es la parte de la máscara que más cubre el centro de abajo, y su punto lejano es la x media de esa parte en sus filas de arriba. Hay camino delante en 71 fotos, y en 30 de ellas hay bifurcación o hueco en el extremo lejano (etiquetas dudosas, `runs/line_follow/labels.jpg`). Las fotos miran casi siempre a lo largo del camino, así que se simula girar la cámara con 7 vistas cuadradas desplazadas por foto, más su espejo. El objetivo es la x del punto lejano respecto al centro de la vista, de -1 a 1. 5 particiones por foto, int8, contra las etiquetas de SAM 3 sin revisar.

| Perfil | Receta | Error medio | Izquierda/recto/derecha | Lado contrario | Giros claros (abs > 0,5) al lado bueno |
|---|---|---|---|---|---|
| Siempre recto | | 0,577 | 16,9% | 0% | |
| ESP32 96 px | última capa | 0,572 | 51,2% | 15,4% | |
| ESP32 96 px | red entera | 0,541 | 55,7% | 14,8% | 76% |
| Móvil/PC 224 px | última capa | 0,435 | 60,5% | 8,7% | |
| Móvil/PC 224 px | red entera | 0,407 | 60,3% | 4,5% | 88% |

- En móvil/PC funciona como prueba de concepto: casi nunca gira al lado contrario (4,5%) y acierta el lado de 88% de los giros claros. En ESP32 a 96 px es flojo (15% al lado contrario).
- Las fotos con bifurcación no puntúan peor que las otras.
- Límites: fotos a altura de persona (un robot ve el suelo de mucho más cerca), 71 fotos, giro simulado con recortes (sin perspectiva real) y verdad de SAM 3 sin revisar.

## Profesores para seguir caminos: comparativa en datasets públicos (2026-10-03)

Pregunta: qué modelo grande etiqueta mejor "el camino va a la izquierda, recto o a la derecha". Mismas tres clases en los dos datasets; scripts en `experiments/line_follow/` (`trails_*.py`, `drive_*.py`), resultados en `runs/line_follow/trails/` y `runs/line_follow/drive/` (hojas visuales `teachers.jpg`).

- **Carretera, Sully Chen dataset 1** (salpicadero de coche, ángulo de volante, MIT): izquierda bajo -15°, recto entre -3° y 3°, derecha sobre 15° (lo de en medio fuera). 3 muestras disjuntas de 150 fotos (50 por clase), separadas al menos 20 fotogramas. Umbrales de los profesores que dan un punto ajustados en las otras dos muestras.
- **Senderos de monte, IDSIA Forest Trails** (Giusti et al. 2016, CC BY 4.0): carpetas de test 011 y 012, 600 fotogramas (200 por clase; la 013 son vídeos de dron sin etiquetas). Etiqueta por cámara: la que mira 30° a la izquierda es "derecha", etc. El `info.txt` de esas carpetas dice que el caminante giraba a propósito: etiquetas con ruido. El artículo da 85,2% a su red y 82-86,5% a dos humanos en su test, y 36,5% al método de segmentar y tomar el centroide.

| Profesor | Carretera (media ± sd) | Senderos |
|---|---|---|
| Molmo2-8B señalando, centro ajustado (x 0,59) | **0,800 ± 0,029** | |
| Molmo2-8B señalando, centro 0,5 | 0,656 ± 0,016 | 0,278 (umbral en el test; no señala en 42%) |
| Qwen3.8-27B (Token Factory) | 0,664 ± 0,027 | **0,543** |
| MiniCPM-V-4.5 | 0,609 ± 0,021 | 0,487 |
| MiniMax-M3 | 0,607 ± 0,024 | 0,517 |
| Kimi-K2.6 | 0,604 ± 0,022 | 0,535 (11% sin respuesta) |
| GLM-5.3-Flash | 0,593 ± 0,034 | 0,532 |
| SAM 3 + geometría (mejor regla) | 0,480 ± 0,033 | 0,277 (umbral en el test; no ve sendero en 49%) |
| gemma-3-27b | 0,382 ± 0,006 | 0,408 |
| Azar | 0,333 | 0,333 |

- Alumnos con las etiquetas reales de senderos (1.350 fotogramas de las carpetas 005, 008 y 009, int8): móvil/PC 61,7% (última capa) y 60,2% (red entera); ESP32 55,8% y 53,3%. El 85,2% del artículo usaba 17.119 fotogramas de entreno y, probablemente, un test más limpio.
- SAM 3 + geometría, el profesor actual de `follow.py`, es el peor en los dos: con "road" pinta toda la calzada (los dos carriles y a veces la acera), y en el monte no adivina cuando el sendero no sale en la foto.
- Molmo2 (Apache 2.0, local, unos 18 GB en bf16, 1,4 s por foto en la 3090) es el mejor en carretera, pero solo con el centro ajustado: el coche va por la derecha y el punto lejano de su carril cae a la derecha del centro. Ese ajuste pide unas pocas fotos etiquetadas del usuario (encaja con la calibración del profesor). En senderos casi no señala.
- Los VLMs de Token Factory adivinan por el contexto cuando el camino no se ve, por eso ganan en senderos. Qwen3.8-27B es el mejor en los dos; GLM-5.3-Flash da casi lo mismo 12 veces más barato (0,22 $ por 1.000 fotos frente a 2,63).
- Parte del error es de las etiquetas: el volante dice lo que hizo el conductor (girar en un cruce sin curva visible) y no siempre dónde va la carretera.
- Descartados: InternVLA-N1 (CC BY-NC-SA, entrenado en interiores simulados, necesita su propio código de navegación).

## Velocidad de entrenamiento de PicoDet-S (2026-10-05)

`experiments/picodet_speed.py` (resultados en `runs/bench/picodet_speed.jsonl`), RTX 3090. Casos: gansos `geese_C` (109 train, 27 val, 80 test con verdad FSC-147) y la mesa `table_count_s300` (10 train, 3 val, 12 test: ruidoso, F1 se mueve 5 puntos entre ejecuciones iguales).

- Desglose de una semilla (900 pasos, épocas de 7 pasos, mesa): entrenar 116 s, export Paddle + ONNX 5 s, TFLite 19 s. Con épocas de al menos 50 pasos entrenar baja a 58 s con los mismos pasos: PaddleDetection reinicia los procesos del cargador en cada época y con épocas de 7 pasos la mitad del tiempo era eso.
- 3 semillas, épocas de 50: en serie 256 s, en paralelo 157 s. Con `taskset` a 8 núcleos (como la L40S de Nebius) 162 s: en local la CPU ya no limita. Convertir a TFLite en hilos de un mismo proceso no gana nada (58 s cada una a la vez, igual que en serie): va en procesos aparte.
- Pasos en gansos (3 semillas, F1 test): 900 0.713, 600 0.714, 300 0.696, 200 0.686. Elegido 600.
- Resultado en el pipeline (`table_count_fast`, `--from train`): etapa train 92 s para 3 semillas, antes unos 430 s. F1 test de la mesa 0.551 +/- 0.017, dentro del ruido de sus 12 fotos (900 pasos: 0.575-0.638); la precisión es baja en todas las variantes (0.38-0.50), no es cosa del cambio.
- Pasos por época (600 pasos, gansos, 3 semillas, F1 test medio): 50 (11 épocas) 0.714, 100 (6) 0.719, 200 (3) 0.705; entrenar 80 / 74 / 69 s en paralelo. Diferencias del tamaño de la dispersión entre semillas. Elegido 100 con un mínimo de 3 épocas (el mínimo de 10 inflaba los pasos con épocas largas).
- Pendiente: medirlo en la GPU caliente de Nebius (antes 675 s).

## Imagen a 3D: SAM 3 + TRELLIS v1, TRELLIS.2 y SAM 3D Objects (2026-10-05)

Mismas 10 fotos de Wikimedia que la prueba de TRELLIS v1. Ahora el recorte lo hace SAM 3 por texto (`experiments/img2mesh/sam3_masks.py`, 0,5 s por foto, entorno `teacher-sam3`) y sustituye a OWLv2 + BiRefNet. Scripts: `run_trellis.py --cutout sam3`, `run_trellis2.py` y `run_sam3d.py`. Hojas lado a lado en `data/img2mesh_test/compare_{bee,cam,wrench}.jpg`.

Válidos a ojo, de 10:
- TRELLIS v1 con OWLv2 + BiRefNet: unos 6. Falla la abeja 1 (láminas), la abeja 2 (deforme) y la cámara 3 (tumbada sobre una lámina); la cámara 1 trae basura flotante.
- SAM 3 + TRELLIS v1: unos 8. Arregla la abeja 2, la cámara 3 (ya no tumbada), la basura de la cámara 1 y el tubo de la cámara 4. Siguen mal la abeja 1 y el soporte de la cámara 2. Mismo coste: 7 s + 9 s, 9,7 GB.
- SAM 3 + TRELLIS.2-4B (1024_cascade): unos 5. Peor que v1 en estas fotos: abeja 3 en trozos, cámara 2 como un bulto, cámara 3 como una cúpula hueca, llave 3 doblada. Tarda 16-88 s por objeto. Ojo: corre con transformers 5 y un parche nuestro para las capas de DINOv3 (`model.model.layer`), que podría influir. RMBG-2.0 (CC BY-NC) se sustituye por un stub, porque nuestros recortes ya traen alfa.
- SAM 3 + SAM 3D Objects: 10. Es la única que saca la abeja 1 y la cámara 2 bien, con alas, patas y soporte. Los colores salen más apagados (colores por vértice). Tarda 10,5-17,5 s y llega a 17-18 GB de pico en la 3090: el requisito de "32 GB" del README no se cumple en inferencia. Además da la pose del objeto respecto a la cámara (rotación, traslación y escala).

Licencias, leídas en el código:
- TRELLIS v1: MIT, pero `to_glb` usa nvdiffrast y mip-splatting, ambos no comerciales.
- TRELLIS.2: MIT. Necesita DINOv3 (DINOv3 License: comercial, con "Built with DINOv3" visible y sin uso militar). `o_voxel.postprocess.to_glb` usa nvdiffrast (no comercial).
- SAM 3D Objects: SAM License (comercial; sin uso militar ni ingeniería inversa; hay que incluir la licencia si se redistribuye). Pesos restringidos en Hugging Face. El notebook de Meta fuerza pytorch3d (BSD), pero solo para muestrear la textura: el horneado (modos fast y opt) y el relleno de agujeros siguen pasando por `utils3d.torch.RastContext`, es decir, nvdiffrast. El render de gaussianas usa por defecto el de Inria (no comercial); gsplat (Apache) funciona con `backend='gsplat'`. Ruta limpia hoy: malla con colores por vértice, sin post-proceso. Sale pesada: 70k-700k caras y 1-13 MB.

Instalación: `tools/install_sam3d.sh` en el SSD Crucial, entorno `/media/lenena-iker/Crucial/envs/eyeforge-sam3d`. El índice `pypi.ngc.nvidia.com` de su setup ya no resuelve: quitarlo de `PIP_EXTRA_INDEX_URL`. `notebook/inference.py` exige `CONDA_PREFIX`.

Sin SAM 3 (máscara de la foto entera, `run_sam3d.py --full-mask`, hojas `compare_mask_*.jpg`): 2 de 10. Reconstruye lo que domina la foto, no el objeto: la flor en vez de la abeja, el seto, la pared o el edificio en vez de la cámara, y la llave con la mesa debajo. SAM 3 hace falta: es el que dice qué objeto reconstruir.

Exportación limpia hecha: `experiments/img2mesh/bake_vertex_colors.py`, solo CPU. Simplifica a 20k caras con pyvista, desenvuelve las UV con xatlas, rasteriza las UV con PIL y colorea cada texel con sus 4 vértices densos más cercanos (KD-tree). Pasa de 1-13 MB a 1-1,4 MB y tarda 2-5 s por modelo; con `closest_point` de trimesh tardaba unos 20 s, sin diferencia a ojo. Los 10 pasan por `glb2usd` sin errores y en Isaac (`isaac_views.py`) se ven bien: textura, escala y eje vertical correctos, y colores más vivos que en pyrender. El objeto sale en la orientación de la foto (la cámara de techo, colgando). Total por objeto: unos 13-22 s (generación más horneado), frente a 17 s de TRELLIS v1.

Pendiente para usarlo en el pipeline: runtime `sam3d` en la etapa `generate_assets` (SAM 3 primero, por texto), su imagen para Nebius y probar fotos del usuario y fotos con varios objetos.

## Escena desde una foto: Mira-Scene (2026-10-05)

Búsqueda previa de modelos que reconstruyen un entorno desde una foto: HY-World 2.0 (Tencent) prohíbe su uso en la UE; Lyra 2.0 (NVIDIA) y SHARP (Apple) tienen pesos no comerciales; SceneGen lleva VGGT-1B (no comercial); PixARMesh es CC-BY-SA y solo interiores. Queda Mira-Scene (VAST, septiembre de 2026, código y pesos MIT): SAM 3 + VLM (lista de objetos y apoyos) + MoGe-2 + CCM propio + SAM 3D por objeto + plano de suelo + colocación con gravedad + panorama de entorno.

Prueba en la 3090 con 4 ejemplos suyos y 4 fotos nuestras (calle, fachada, botellas, escritorio). La lista de objetos de nuestras fotos la escribí a mano (`experiments/scene3d/cases.json`, en lugar del VLM) y SAM 3 hace las máscaras (`mira_prepare.py`). Sin panorama ni relleno del suelo (piden su API de pago). Hojas con la foto y vistas nuevas (`render_scenes.py`) en `data/scene3d_test/sheets/`.

- Objetos pequeños y medianos: muy bien (sillas, mesa, bici, botellas, tapones), apoyados en el suelo y aguantan girar la cámara 90°.
- Edificios y fondos: no salen. La calle queda en moto, coche y señales sobre un suelo de 20 m; sin suelo visible (fachada) no monta la escena. En el escritorio se pierden los tubos.
- El suelo lleva manchas donde estaban los objetos (el relleno es la API de pago).
- Coste: profundidad 7 s en total, CCM 1,7 min por escena, SAM 3D 3 s por objeto (solo la etapa 2, porque el CCM da los vóxeles), escena 4 s.
- Instalación: Crucial `tools/Mira-Scene`, `tools/MoGe`, entornos `tools/envs/mira-{geometry,ccm}` (conda-forge les puso PyTorch sin CUDA o se atascó resolviendo: PyTorch cu121 por pip a mano), mallas con nuestro entorno `eyeforge-sam3d` y un parche en `3_inference_mesh.py` para colores por vértice sin nvdiffrast.

Decisión (usuario, 2026-10-05): primero construir nuestra versión para el fondo (composición sobre la foto real con profundidad, o planos con la textura de la foto). Si sale muy bien, se usa siempre la nuestra; si sale mal, Mira-Scene solo cuando haya suelo visible y objetos grandes, y si no, nada.

### Nuestra versión del fondo, primera prueba (bottles, 2026-10-05)

Scripts en `experiments/scene3d/`: `bg_prepare.py` (SAM 3 + LaMa borra los objetos), `bg_depth.py` (MoGe-2), `bg_build.py` (plano de la mesa, medidas, malla del fondo, réplica de comprobación y vistas nuevas). Resultados en `data/scene3d_test/ours/bottles/` (`sheet.jpg`, `report.json`, `scene.glb`).

- Geometría relativa buena: las mallas de SAM 3D recolocadas donde estaban los objetos coinciden con la foto (IoU de silueta 0,76-0,90, error mediano de profundidad 0,4-3 %). La mesa sale plana a ±2 mm.
- Las mallas de SAM 3D salen demasiado anchas (ancho/alto 0,52-0,64 contra 0,31-0,45 reales): se escalan con el ancho medido en la foto.
- Escala absoluta: con la cinta del usuario (botellas sin tapa 21,5, 22 y 15,5 cm), MoGe-2 da 24,7, 24,2 y 18,7 cm (de +10 a +21 %). Mira-Scene daba 15,9, 17,1 y 16,1 (de -26 a +4 %). Un factor único (0,908, cámara a 0,555 m en vez de 0,611) deja 22,4, 22,0 y 17,0. Ese factor está ajustado con las propias medidas, así que en producción hace falta un dato conocido (altura de la cámara o tamaño de un objeto, de la entrevista): opción `--camera-height`.
- El FOV del EXIF (45,9°) frente al de MoGe (47,0°) cambia poco.
- Una sola foto solo cubre su encuadre: la cámara puede girar unos ±20° y acercarse, no alejarse. Las vistas con más de un 8 % sin fondo se descartan.
- Pendiente: luz y sombras de los objetos (Isaac Sim) y objetos tumbados.
- Isaac Sim (`bg_isaac.py`): el fondo es la malla con la foto borrada como textura; cada vista se renderiza con objetos (A) y sin ellos (B). El fondo final es la foto vista desde esa cámara por A/B (las sombras de los objetos caen sobre la foto real) y los objetos toman A con la ganancia que lleva B a la foto. El color de la mesa sale igual que en la foto (106 contra 105 de media). Los colores de vértice de glTF son sRGB y `displayColor` es lineal: sin convertirlos, los objetos salen lavados.
- Defectos que quedan, de las mallas y no del montaje: SAM 3D cierra la boca de la botella negra grande y deja la blanca beis y sin brillo.
- Color (consulta del usuario: el plástico blanco y las líneas blancas salen oscuros). Brillo mediano de la botella blanca: foto 162, colores de SAM 3D sin luz 137, Isaac 80. Oscurecen las dos etapas: SAM 3D un 15-40 % y, sobre todo, la luz de Isaac (sol casi vertical; los laterales reciben la mitad que la mesa). Arreglo automático sobre la réplica (vista 0):
  - Luz: búsqueda en rejilla (25 combinaciones de altura, azimut y fuerza del sol) que compara las sombras reales (foto / fondo borrado, en la mesa) con las de Isaac (A/B). Puntuar por el sombreado de los objetos no sirvió: los logos mal alineados lo dominan y eligió un sol duro.
  - Color: una ganancia de brillo por objeto, 3 iteraciones. Por canal teñía de azul con el balance de blancos de la foto.
  - Resultado: luz suave desde la izquierda; botella blanca 0,40 en la foto contra 0,32 en la réplica (lineal), antes 0,13. El albedo topa en 1, así que el resto pediría más exposición.
- Tonemapper (pregunta del usuario): con la luz fija, cambiar el operador de RTX (`/rtx/post/tonemap/op`, por defecto 6 = ACES) a 1 (lineal) da números idénticos y ningún píxel quemado, así que no era la causa. La causa era el sol: solo con cielo, la botella blanca da 0,404 contra 0,397 en la foto; con el sol elegido antes, 0,317. La búsqueda de luz ahora puntúa las sombras más el brillo que no se alcanza tras ajustar el color, y elige cielo más un sol débil y bajo desde detrás de la cámara (blanca 0,383 contra 0,397). La pasada completa tarda 98 s (37 luces y 7 vistas).

### Generalización: ejemplo de Mira 005_cafe_patio (2026-10-05)

Misma cadena, sin ajustes a mano salvo las frases de SAM 3 (`bistro chair, cafe table, patio umbrella, umbrella base, bicycle`, soporte `sidewalk`). Resultados en `data/scene3d_test/ours/005_cafe_patio/`.

- Lo que generaliza: borrado y fondo, profundidad, réplica (objetos en su sitio y a escala plausible: sillas 0,93-0,99 m, mesa 0,81 m, cámara a 1,41 m; MoGe parece pasarse de nuevo un ~8-10 %) y color de los objetos en Isaac (mimbre, sombrilla beis).
- Arreglos que hicieron falta, ya generales:
  - Orientación por objeto (giro, ancho y desplazamiento) ajustada a la silueta, porque sillas y bicis no son simétricas. En botellas sube la IoU de 0,76-0,89 a 0,89-0,90.
  - Cámaras movidas girando alrededor del centro de los objetos y con límite en metros (`--max-move`, 0,4 m). El eje óptico casi horizontal cortaba el suelo a 22 m.
  - Plano lejano de 50 m a 2 km: la calle llega a 170 m y el recorte salía como un abanico de colores. El cielo sin profundidad pasa a ser un telón lejano.
  - `--shadows` en bg_prepare: SAM 3 'shadow', solo las sombras conectadas a los objetos, se borran con ellos. Si no, quedan sombras sin dueño en el fondo.
- Lo que NO generaliza todavía: la luz con sol duro. La búsqueda elige sol desde detrás de la cámara cuando en la foto viene de delante (contraluz). Causa: la sombra "real" (foto / fondo borrado) no es fiable aquí. La zona borrada es grande y LaMa la rellena más oscura que la foto (126 contra 139 dentro de la máscara de sombra), así que la referencia no contiene las sombras. Comparar por bloques no lo arregla. Mapas en `shadow_maps.jpg` (foto, real, sintética con sol delante, sintética con sol detrás).
- Propuesta: dirección del sol por geometría de las sombras (base del objeto → punta de su sombra en el plano da el azimut, y longitud / altura la elevación), con las máscaras de sombra por objeto que ya tenemos. Más barato y más robusto que la búsqueda.
- Sol por geometría de las sombras (implementado, `estimate_sun` en bg_build y su uso en bg_isaac):
  - Cada píxel de sombra va al objeto más cercano y se proyecta al plano. Base → punta da el azimut, altura / longitud la elevación, y se toma la mediana entre objetos.
  - La fuerza del sol se elige para que la sombra sintética oscurezca el suelo como la real. La real se mide en la foto (sombra / suelo iluminado alrededor), sin LaMa. La sintética, relativa a un render sin sol, para no contar el cielo que tapa la sombrilla.
  - Café: sol de frente y algo a la izquierda, elevación ~34° (6 objetos); oscuridad real 0,70, sintética 0,68 con fuerza 0,35.
  - Sin sombras detectadas (botellas) se usa la búsqueda de antes (blanca 0,385 contra 0,39).
  - El color se ajusta ahora solo en los píxeles donde cada objeto queda delante en la réplica. Antes una silla mal encajada recibía el color de otro objeto.
  - Queda: mesa y sombrilla siguen más claras que en la foto (0,70 contra 0,51) porque saturan a 255. Las sombras sintéticas salen más suaves que las reales.

### HOPE para medir si nuestra versión ayuda al alumno (2026-10-06)

- Descargado en `data/hope/` (861 MB): val y test reales de RealSense y los 28 modelos 3D exactos. Sin `train_pbr` (100 GB), que sería otro brazo (BlenderProc con los modelos exactos).
- Licencia CC BY-NC-SA 4.0: solo para pruebas internas, como MIO-TCD.
- 50 escenas, 10 entornos de 5 escenas consecutivas, ~5 fotos por escena con distinta luz o vista, entre 11 y 26 objetos por foto (mediana 18). El test no trae poses, pero sí `mask_visib` por instancia (coinciden con los totales de `test_targets_bop19`), así que hay cajas sin clase para todas.
- Separación en `data/hope/split.json`: por entorno, la primera escena a train (50 fotos) y las otras 4 a test (188).
- Tarea: contar o detectar "producto" sin clase.
- Riesgo: mucho desorden (cajas, bolsas, montones), más difícil que bottles para SAM 3D y para colocar objetos sobre el plano.

Primera escena por el pipeline (test/000021, mesa con mantel, 2026-10-06; `data/scene3d_test/ours/hope021/`):
- SAM 3 con "can, box, bottle, carton" encuentra 17 objetos, los mismos que la verdad. Mallas SAM 3D para 15. Unos 10 min por escena (CCM 4,6 min, SAM 3D 2,5, Isaac 2,6).
- Escala: la profundidad de la RealSense entre la de MoGe da una mediana de 0,92. MoGe se pasa un 8 % (en bottles, un 10 %): parece un sesgo estable, corregible con un factor fijo.
- Color y luz en Isaac bien: brillo por objeto igual que la foto y nada quemado.
- Falla con el desorden. Los objetos apilados se apoyan en la mesa y el montón se esparce en las vistas nuevas. Las cajas tapadas salen como cáscaras abiertas, porque SAM 3D solo ve una parte. Réplica con IoU entre 0,38 y 0,83 (bottles: 0,76 a 0,90).

Apilados y mallas incompletas (2026-10-06):
- `bg_build.py` detecta el objeto que descansa sobre otro. Columna a columna, mira justo debajo del borde inferior de la máscara: si hay otro objeto que no está claramente delante, se apoya en él y la base sube a la altura de sus puntos más bajos. En hope021 lo detecta en una caja, cuya réplica pasa de IoU 0,38 a 0,64.
- `--stack`: probabilidad de poner un objeto encima de otro en las composiciones nuevas, solo sobre objetos con tapa plana (`flat_top`: una lata tumbada no vale, una caja sí).
- Solo se recolocan los objetos con buena réplica (`--min-iou 0.5`) y visibles en la foto (`--min-visible`). Con 0,6 entran 11 de 15; con 0,7, solo 6.
- Las latas tumbadas siguen saliendo abiertas por los extremos. Causa probable: con Mira solo corremos la etapa 2 de SAM 3D sobre los vóxeles del CCM, que solo ven la parte visible. Lo siguiente es probar SAM 3D completo, etapas 1 y 2 (`experiments/img2mesh/run_sam3d.py`), que inventa la parte oculta.
- El fondo borrado por LaMa deja manchas grises donde estaba el montón, visibles en las vistas nuevas.

Cierres, fondo y sol (2026-10-06, hope021, bottles y café):
- Las mallas abiertas se cierran con tapas sacadas de la envolvente convexa (`close_shell`): solo se añaden las caras de la envolvente lejos de la malla, con el color del vértice más cercano. Sin aviso solo se tapa si el objeto es casi convexo (cubre ≥ 40 % de la envolvente y las tapas ≤ 35 %), lo que deja intactos sillas, bici, mesa y escritorio. Una lata vista a medias no supera esa prueba (se parece a un escritorio), así que `--solid "can,box,..."` marca las frases de objetos cerrados. En el producto lo diría el VLM.
- SAM 3D completo (etapas 1 y 2) cierra bien las cajas, pero convierte las latas en tazas con asa. Descartado por ahora.
- Visibilidad para recolocar: ahora es la parte de la silueta de la réplica no tapada por otros objetos (antes era el IoU con la propia máscara, que castigaba sillas y bicis). `--min-iou` baja a 0,15, porque los objetos finos encajan entre 0,2 y 0,5 aunque estén bien. Café: 5 de 7 objetos; HOPE: 10 de 15; bottles: 6 de 6.
- Fondo: las manchas grises eran sombras de contacto del montón. Con `--grow 0.03` (ahora por defecto) y `--shadows` desaparecen; LaMa a menor resolución no ayudaba.
- Grietas negras en la mesa: los píxeles borrados que la máscara de la mesa deja abiertos al exterior no se pegaban al plano. Ahora se pegan si están dentro del contorno convexo del soporte, y el borde recto de la mesa de bottles sigue bien.
- Sol desde sombras solo si las direcciones coinciden (acuerdo ≥ 0,85): café 0,96 (sol), HOPE 0,71 (sombras de contacto en interior, que antes daban un sol falso con un 4 % de píxeles quemados).
- Hojas en `data/scene3d_test/ours/{hope021,bottles,005_cafe_patio}/sheet_isaac.jpg`.

### HOPE: el alumno con nuestras escenas (2026-10-06)

Las 10 escenas de train por el pipeline (`experiments/scene3d/run_hope_train.sh`, ~12 min cada una, 20 vistas por escena; hope021 de antes con 6). Soportes escritos a mano por escena; objetos `can,box,bottle,carton` con `--solid` y `--stack 0.2`. Cajas de las vistas con `view_labels.py` (mapa de instancias con pyrender; se quitan los restos sueltos de malla, que estiraban cajas). SAM 3D se quedó sin memoria en un objeto de test/000036: `bg_build` ahora salta mallas que faltan y el driver usa `expandable_segments`.

Revisión a ojo (hojas con foto, vistas y cajas):
- Buenas: val/000006, test/000006, test/000031. Regulares: val/000001 (sofá: objetos flotando delante del respaldo), test/000016 (cartones enormes en vistas cercanas), test/000036 (alféizar).
- Malas: test/000001 (objetos en bolsas sobre el suelo; réplica mal y sobreexpuesta), test/000011 (caja de cartón abierta tomada como producto), test/000026 (dos baldas: mallas estiradas).

Alumno PicoDet-S 320 del producto (`hope_student.py`), 3 ejecuciones por brazo, test 188 fotos reales, umbral elegido en val:

| Brazo | Imágenes train | F1 | Error de cuenta |
|---|---|---|---|
| A: solo reales (40) | 40 | 0,819 | 2,97 |
| Q: reales + sintético revisado (7 escenas) | 173 | **0,830** | **2,55** |
| C: reales + todo el sintético | 235 | 0,814 | 3,54 |
| DQ: solo sintético revisado | 106 | 0,761 | 3,45 |
| D: solo sintético, todo | 156 | 0,744 | 5,84 |

- Con revisión, el pipeline ayuda (+1,1 puntos de F1 y -14 % de error de cuenta); sin ella, empeora. Las escenas malas restan más de lo que suman las buenas: el control de calidad es parte necesaria del pipeline.
- Solo sintético, desde 7 fotos sin etiquetar, llega a 0,76 contra 0,82 con 40 fotos etiquetadas a mano.
- Diferencias pequeñas con 3 ejecuciones (desviación ~0,006): A contra Q es indicativo, no concluyente. El brazo B (sintético actual con HDRI y Objaverse) sigue pendiente.

### VLMs de Token Factory como coordinadores (2026-10-06)

Papel (usuario): no sustituyen a MoGe; ayudan a fijar la escala y hacen lo que Claude hacía a mano (frases, soporte, `--solid`, decidir si la escena encaja, revisar lo generado). Kimi-K3, GLM-5.3-Flash, DeepSeek-V4.1-Flash y Qwen3.8-27B (Cosmos3 y qwen3-vl-32b no están en nuestro catálogo). Coste total de céntimos.

`experiments/vlm_bench/vlm_scene.py` (bottles, café, hope021, hope val1; foto limpia + foto con objetos numerados):
- Frases para SAM 3: en HOPE empatan todos (17/17 en hope021). En el café Kimi-K3 da justo las 5 frases a mano (IoU 0,98); los demás añaden jardineras, carteles, persona o coche. En bottles Qwen saca los 8 objetos; Kimi y DeepSeek pierden la cafetera (0,81). Soporte bien salvo GLM en el sofá ("armchair seat", SAM 3 no encuentra nada).
- Tamaños: en las botellas reales, escala de +2 % (GLM) a +21 % (Qwen); todos dan 20-24 cm a la botella corta de 15,5. HOPE son juguetes (latas de 7 cm, cajas de 16,5): todos se pasan 1,2-1,4x, también Claude. Sirven como comprobación gruesa de escala (detectar un 2x), no como corrección fina.
- Respuestas cortadas con 8k tokens (Qwen en bucle, GLM incompleto): con 16k bien.

`experiments/vlm_bench/vlm_qc.py` (las 9 escenas de arriba, verdad = revisión a ojo):
- Antes de generar, solo con la foto, "¿cabe todo en un único soporte plano?": los cuatro aciertan 9/9 (sofá, bolsas, caja de cartón y dos baldas detectados). Este filtro previo funciona ya.
- Revisión de las hojas generadas: los cuatro detectan las 3 malas (Kimi 2/3) pero son mucho más severos que la revisión a ojo y llaman "mala" a casi todo (escenas aprovechables aceptadas: GLM 3/6, Kimi y DeepSeek 1/6, Qwen 0/6). Los defectos que citan son reales (caja de zumo rota en test/000031, objetos sin sombra de contacto). Como juez binario descartaría casi todo; hace falta otra forma de preguntar (puntuación o comparación entre escenas) o medir con el alumno qué severidad conviene.

### El VLM coordina: plan previo y revisión con nota (2026-10-06)

- `experiments/scene3d/bg_plan.py` (paso 0, Kimi-K3 por defecto, `SIM2CHIP_PLAN_MODEL`): una llamada con la foto da soporte plano único sí/no, frase del soporte, frases de objetos y lista `--solid`. Con la llamada combinada sigue 9/9 en HOPE, y bottles y café pasan. Pegas: en el café añade "car" (el coche aparcado se borraría) y pierde la base de la sombrilla; en test/000006 añade "paper sheet". El driver (`PLAN=1 run_hope_train.sh`) salta las escenas sin soporte único (test/000026: "split across two separate shelf boards") y genera las demás con el plan: val/000006 sale con 21 objetos, igual que con las frases a mano.
- Revisión con nota de 0 a 10 (`vlm_qc.py rate`), avisando de que el aspecto sintético es normal y solo cuenta lo que enseña mal al detector. AUC (escenas aprovechables por encima de las malas): Qwen3.8-27B 1,00 (malas 1-2, aprovechables 3-4), GLM-5.3-Flash 0,83, Kimi-K3 0,78. DeepSeek-V4.1-Flash gasta los 16k tokens pensando y no responde.
- `experiments/scene3d/bg_review.py` (último paso, Qwen3.8-27B, `SIM2CHIP_REVIEW_MODEL`, umbral 3): hoja con la foto y 5 vistas con cajas, `review.json` con nota y `keep`. Rechaza justo las 3 escenas malas de la revisión a ojo, así que el brazo R de `hope_student.py` (reales + escenas que el VLM acepta) tiene las mismas 133 imágenes que Q: F1 0,830 sin revisión humana.
- Margen estrecho: tres escenas aceptadas sacan justo 3 (hope021, val/000001 y val/000006 planificada), y las notas cambian un punto al cambiar las vistas de la hoja. Con 9 escenas el umbral es frágil; hace falta más escenas para fijarlo.

### HOPE: brazo B, el sintético actual del producto (2026-10-06)

Spec `specs/benchmarks/hope_count.json`, escrita como la saldría de la entrevista: `describe-scene` sobre las 50 fotos de train propone interior, mesa de madera y cámara a 0,5-1 m. Clase "product" con 6 búsquedas (lata, caja de cereales, bote de ketchup, cartón de leche, zumo, pasta). El job `data/jobs/hope_count` corre las fases del producto hasta `points`: 60 modelos de Objaverse, GLM-5.3-Flash deja 30, y 2000 imágenes en 3,5 min, con 3-4 objetos sobre un mueble en una habitación. Mismo alumno y mismo test que los demás brazos (F1 medio de 3 ejecuciones, error de conteo):

| Brazo | Imágenes de entrenamiento | F1 | Error de conteo |
|---|---|---|---|
| A, solo reales | 40 | 0,819 | 2,97 |
| BS40 | 40 + 40 del producto | 0,827 | 2,62 |
| BS, tantas como Q | 40 + 133 del producto | **0,833** | **2,47** |
| Q, nuestras escenas revisadas | 40 + 133 nuestras | 0,830 | 2,55 |
| BSQ | 40 + 133 + 133 | 0,822 | 2,82 |
| BS300 | 40 + 300 | 0,817 | 2,73 |
| BS600 | 40 + 600 | 0,809 | 3,07 |
| B, lo que haría el producto hoy | 40 + 2000 | 0,776 | 3,61 |
| BW, B con las reales repetidas (real_weight 4) | 40 x 12 + 2000 | 0,818 | 2,65 |
| BQ | 40 + 2000 + 133 | 0,779 | 3,69 |

- Por imagen, el sintético del producto rinde igual que nuestras escenas desde la foto en HOPE: 0,833 frente a 0,830, dentro del ruido (~0,006). Nuestra ventaja visual no se nota en este alumno ni en este dataset.
- Lo que manda es la dosis: con 40 reales, entre 40 y 133 sintéticas ayudan, y a partir de ~300 restan. 2000, el valor por defecto, le quitan 4 puntos al alumno.
- Causa probable: PicoDet entrena 600 pasos fijos (`sim2chip/detector.py`), así que con 2000 sintéticas las fotos reales se ven 12 veces menos. `training.real_weight` ya repite las reales en la ruta de la cabeza de conteo, pero no en la del detector. Aplicándolo (BW) se recupera el nivel de solo reales (0,818) y baja el error de conteo (2,65), pero no llega a lo que dan 133 sintéticas.
- Mezclar las dos fuentes no suma (BSQ 0,822): otra vez, más sintético con las mismas 40 reales.
- Propuesta para el producto, pendiente de confirmar en otro dataset de conteo antes de tocarlo: en el detector, limitar el sintético a unas 3 veces las reales, o al menos aplicar `real_weight`.

### Cuánto sintético por foto real: literatura y decisión (2026-10-06)

- El óptimo depende de la tarea y del detector. Lo publicado va de 0,5 a 4 sintéticas por real: 8:2 en Kim et al. y 1:0,5 en un estudio de conducción rural ([2607.27058](https://arxiv.org/pdf/2607.27058)). Con detectores clásicos de 2014 llegaba a 15-20:1 ([1411.7911](https://arxiv.org/pdf/1411.7911)).
- Varios estudios ven que con demasiado sintético las reales quedan ahogadas y el rendimiento baja ([2211.16066](https://arxiv.org/pdf/2211.16066)). En otro, pasar del 25-50 % de sintético empeora en todos los pipelines ([2412.01477](https://arxiv.org/pdf/2412.01477)).
- Preentrenar con sintético y afinar con reales suele ir mejor que mezclar ([1907.07061](https://arxiv.org/pdf/1907.07061)).
- En la detección de palés ([2510.12208](https://arxiv.org/html/2510.12208v1)), el sintético muy parecido al real mejora dentro del dominio pero empeora fuera de él, y el aleatorio hace lo contrario. Encaja con la idea del usuario de variar los fondos.
- Decisión del usuario: no limitar. La web colorea las opciones de imágenes sintéticas (verde hasta 4 por foto real, amarillo hasta 10, rojo por encima, solo en conteo con fotos) y el usuario decide (`web/src/pages/Interview.tsx`).

### Escenas desde una foto: muebles grandes y lo que copiamos de Mira (2026-10-06)

Comparación con Mira-Scene en bottles, café y la oficina (`003_home_office`), hojas en `data/scene3d_test/compare/`. En bottles y café gana lo nuestro: Mira deja agujeros grises y fondo negro porque no ejecutamos sus etapas 4 y 6, que piden la API de imagen de Tripo. En la oficina ganaba Mira con claridad.

Cómo construye Mira la escena (`infer_scripts/5_construct_scene.py`):
- Objeto a objeto: el CCM da, por píxel, el punto de la malla canónica que se ve. Un ajuste RANSAC de similitud contra la profundidad saca giro, escala y posición, aunque el mueble esté cortado o tapado. Después un grafo de apoyos del VLM asienta cada objeto sobre su padre, sin física.
- No guarda la foto como fondo. El suelo es un plano con textura (recortada de la foto o generada), y el resto es un panorama 360 generado por GPT Image a partir de la foto sin objetos ni suelo.

Por qué fallaba lo nuestro en la oficina (medido):
- Al borrar con LaMa un escritorio o una estantería enteros queda una mancha gris. MoGe la ve como una pared a ~1 m, con los muebles a 2,5-4,5 m, y el fondo los tapaba. La réplica bajaba a 0-0,15 de coincidencia de silueta y se descartaban, aunque el ajuste solo daba 0,52-0,75.
- MoGe sobre la foto con huecos grandes deforma la habitación entera: paredes con más de un 5 % de error frente a la foto.
- El ajuste por silueta no distingue delante de detrás: la estantería salía de espaldas, como un panel liso.

Cambios en `bg_build.py`:
- El fondo usa la profundidad de la foto donde no se ha borrado nada.
- Los huecos fuera del soporte toman la profundidad de la pared que los rodea, con un relleno armónico (`smooth_fill`; `cv2.inpaint` en coma flotante mete ruido que rompe la malla). Dentro de la máscara de cada objeto se empujan detrás de su malla, con una rampa suave en el margen.
- `--fixed`: los muebles (los marca el VLM en `bg_plan.py`, campo `fixed`) se quedan en su sitio en todas las vistas y tapan su propio hueco. Los demás objetos se mueven alrededor.
- Orientación desde el CCM de Mira (`bg_ccm_pose.py`, reutiliza su solver de la etapa 5): el ajuste por silueta solo busca a ±30° de ella.
- `--layout photo`: los objetos se quedan donde están en la foto y solo se mueve la cámara (copiar la escena en vez de generar variedad). En `run_photo.sh` con `LAYOUT=photo`.

Resultado en la oficina: coincidencia de 0,63 en el escritorio, 0,50-0,73 en las estanterías y 0,63 en la silla. La estantería sale de frente con sus libros, sin mancha delante, y 20 vistas aceptadas con la silla y la lámpara movidas. Quedan defectos de las mallas de SAM 3D (escritorio como bloque, panel lateral en la estantería) y algo de mancha en el suelo donde estaba el escritorio. En bottles y café no cambia nada: el error de fondo frente a la foto baja a cero. El plan del VLM con el campo `fixed` acepta ya la oficina (escritorio, estantería y armario fijos) y en café, bottles y HOPE da lo mismo que antes.

Escena por piezas (idea del usuario, 2026-10-06). La foto ya no se usa como fondo: suelo, soporte y paredes son planos en Isaac ajustados a la profundidad de la foto original, y cada uno lleva un material recortado de la foto.
- `piece_surfaces.py`: SAM 3 recorta "wall" y "floor".
- `piece_build.py`:
  - Ajusta los planos: el soporte por RANSAC; las paredes verticales por RANSAC en coordenadas del suelo; el suelo, paralelo al soporte si está debajo.
  - Rectifica cada superficie como vista de frente y recorta el mayor rectángulo limpio. Lo hace repetible fundiendo los bordes (en espejo dibujaba rombos con los tablones).
  - Cierra la habitación con las paredes que no se ven y un techo. Con `--indoor`, también sin paredes visibles.
  - Coloca los objetos según la foto (`photo_replica`, orientación del CCM) y genera cámaras libres alrededor.
- `piece_isaac.py`: render 100 % sintético. Interior con panel de techo y lámpara; exterior con HDRI y sol. Exposición automática en los dos sentidos. Etiquetas con `view_labels.py`.
- Los muebles fijos ya no se tapan con el casco convexo: la estantería abierta salía con triángulos.

Revisión del VLM (Qwen, 0-10, se descarta por debajo de 3):

| Escena | Fondo de la foto | Por piezas |
|---|---|---|
| Oficina | 1 (2 con los parches de huecos) | 3 |
| Botellas | 2 | 5 |

- Quedan defectos de las mallas de SAM 3D: escritorio cortado por el borde de la foto, que sale como un bloque estrecho, y brazos de lámpara estirados.
- El recorte del tablero de las botellas es pequeño y lleva un degradado de luz que se nota al repetirlo.
- El café queda fuera de las pruebas: la foto es generada y de mala calidad (usuario). La de la oficina es un ejemplo de Mira y probablemente también es generada; falta una foto real de una habitación.

Street corner frente a Mira (2026-10-06), `data/scene3d_test/compare/006_street_corner.jpg`. Es la misma foto vista desde su cámara y girada ±20° y ±40° alrededor de los objetos: `piece_build.py --orbit` para lo nuestro y `mira_views.py` para su `scene_with_floor.glb`.
- Modo exterior nuevo (`--outdoor`): cada fachada es una pared del tamaño que muestra la foto, sin cerrar habitación, con HDRI de cielo. La tolerancia del RANSAC de paredes pasa a ser relativa a la distancia: con 3 cm fijos no aceptaba fachadas a 7-10 m.
- Mismos 5 objetos y la misma disposición en los dos. Mira pone un suelo con textura generada que no se parece a la acera y deja todo lo demás en negro (no ejecutamos su etapa de entorno). Lo nuestro tiene fachadas de ladrillo y muro bajo con el material de la foto, y una acera gris plana.
- Fallos nuestros:
  - Las fachadas salen lisas, porque el recorte limpio quita las ventanas.
  - Los HDRI disponibles son de campo y no pegan con una calle.
  - No hay calzada ni bordillo.
  - Con algún sol el suelo sale quemado.

### Entorno con SAM 3D (street_corner, 2026-10-06)

`env_add.py`: casa, seto y muro como "objetos" de un caso de Mira (`mira_prepare.py` con frases "house", "hedge", "low brick wall"; etapas 1-5, la 4 solo geométrica), y sus mallas se añaden fijas a nuestra escena por piezas en lugar de las fachadas planas (`--drop-walls`). Escala de la profundidad de Mira a la nuestra: 1.004. Hoja: `data/scene3d_test/compare/006_street_corner_env.jpg` (Mira, nuestro con planos, nuestro con SAM 3D) y mallas sueltas en `ours/street_env/env_meshes.jpg`.

- Las casas salen muy bien: la principal es un bloque con ventanas en las dos caras, la del fondo con tejado y chimenea, la blanca con su anexo. Coste: unos 2 min de CCM y SAM 3D por instancia en la 3090 (a 80 C).
- Seto principal y arbusto suelto, aceptables (bloque verde con relieve). El muro bajo, una losa correcta.
- Fallan instancias pequeñas o mezcladas: un "hedge" que SAM 3 juntó con la casa de detrás (sale medio casa, medio seto) y un pilar de 1 m.
- Lo que más estropea ahora es el fondo: HDRI de roca o desierto donde no hay casa, y la acera sin calzada ni bordillo.
- HDRI "infierno": era `white_cliff_top`/roca de `data/hdri` (todos de campo), y además tumbado 90°: `piece_isaac.py` alineaba el polo de la cúpula con su +Y, pero en RTX es +Z. Corregido; 7 HDRI urbanos diurnos de Poly Haven (CC0, 2k) en `data/hdri/urban` (`--hdri-dir`).
- `env_add.py --max-faces 20000` (open3d, decimación cuádrica con colores): `meshes.npz` del entorno de 42 MB a ~1 MB; la escena entera 19 MB, casi todo los 5 objetos sin simplificar. El seto se ve algo facetado a 20k.
- Vistas "apagadas" (0, 2, 3): no era el sol (ningún mesh del entorno lo tapa, comprobado con rayos) sino la exposición. `expose()` llevaba cada vista a mediana 0.42: con sol de frente todo queda iluminado por igual, nada en sombra compensa y la imagen entera baja a gris; con sol de lado (1, 4) la mitad en sombra deja lo iluminado muy claro. Ahora expone por las luces altas (p97 a 0.92) con la mediana entre 0.32 y 0.58: las cinco vistas igual de vivas. Queda el suelo casi blanco al sol (albedo gris 0.52 de la acera). Sin probar aún en oficina y botellas.

### De una esquina a un cruce (`crossroad.py`, 2026-10-06)

La esquina como bloque en una cuadrícula: ejes desde la normal de la fachada, esquina en el bordillo (percentil 1 de la acera en esos ejes, sobre la profundidad de la foto). Copias giradas 90/180/270 grados alrededor del centro del cruce (giro, no espejo: textos y bicis no se invierten), ladrillo distinto por copia (tono, saturación y brillo de los vértices naranja-rojo; ventanas, tejados y plantas igual), calzada de asfalto 12 cm por debajo de las aceras con caras de bordillo. Entorno lejano asentado en la acera (la casa blanca flotaba 4 m). Cámaras en calzada o junto al bordillo, a más de 7 m de cualquier superficie, apuntando a una esquina ±35 grados, 1.3-4 m de alto. Hoja: `ours/street_cross/sheet_isaac.jpg`.

- Se ve como un cruce real en la mayoría de vistas; el giro de 90 grados no se nota a pie de calle.
- Pendiente: las casas de SAM 3D están huecas por arriba (solo se ve desde la vista cenital); los mismos 5 objetos en cada esquina (barajar por esquina); 72 MB por copiar las mallas 4 veces (instanciar el bloque y simplificar los objetos lo bajaría a pocos MB).
- Peso: `--no-objects` (solo el lugar; los objetos los pondrá Replicator) y entorno lejano (> 25 m de la esquina) como cajas con una foto de 24 px por cara (render ortográfico sin luz de la malla) o fuera del todo (`--far-mode drop`, queda el HDRI). De 72 MB a 3.6 MB de mallas + 164 KB de texturas. Desde la calle las cajas pasan por casas lejanas, algo pálidas.
- Qué es automático aquí: SAM 3, MoGe, CCM, SAM 3D, colocación, materiales rectificados, bordillo, copias, ladrillos, cámaras, exposición. A mano: las frases del entorno, `--outdoor`, el grupo de HDRI urbano, `--drop 10,12` y elegir el cruce. Kimi y Qwen no han intervenido en street_corner (no hay plan.json).

### Flujo de escena en tres fases sobre fotos reales de calle (2026-10-07)

Banco: 9 fotos de Wikimedia Commons (`data/street_test/`, licencias en `sources.json`): 7 calles de pueblo de día a la altura del peatón (esquinas, calles rectas, una fachada con coche) y 2 de noche para probar el enrutado. Orquestador `experiments/scene3d/photo2scene.py` (antes `scene_flow.py`): cada subtarea es un script con salida en disco, su comprobación y su registro en `flow.json`; se reanuda donde se paró y `--redo X` repite X y todo lo que depende de él.

Fase 1 (Extracción):
- 1.1 `flow_observe.py`: Kimi-K3 con la ficha de capacidades (`flow_capabilities.md`) y el catálogo de fallos (`flow_failures.md`), observación antes de decidir, 3 muestras votadas. Rutas bien en 9/9 (las dos de noche fuera de alcance), los tres votos coinciden siempre. Las notas por entorno ("cortado por el borde", "puede fundirse con la casa de detrás") son útiles. La altura de cámara del VLM es siempre 1,6 m: la mide MoGe.
- 1.2 `flow_route.py`: ruta, plantilla y grupo de HDRI por reglas en código. Calles de pueblo y plazas con cielo `countryside`: el urbano ponía bloques de pisos detrás de una iglesia.
- 1.4 `flow_camera.py`: plano de suelo en la foto borrada; altura 1,3-2,2 m, alabeo < 3 grados y horizonte a menos de 0,07 del que dijo el VLM en las 7.
- 1.3 `flow_segment.py` + `flow_check_masks.py`: SAM 3 del entorno sobre la foto sin coches; puertas de tamaño y borde en código, y preguntas sí/no al VLM. Contra mi revisión a ojo (33 candidatas): Qwen3.8-27B 30, Kimi-K3 29, Gemma-3-27B 27, GLM-5.3 no responde. Qwen a temperatura 0 cambia de respuesta entre ejecuciones: 3 muestras votadas. Reparación (el VLM nombra lo que sobra, SAM 3 lo recorta) y segunda opinión de Kimi antes de tirar una instancia grande; una instancia grande con solo "se lleva otra cosa" pasa con marca (perder la casa principal cuesta más que 2 min de SAM 3D).
- 1.5 `flow_mira_case.py`: la foto se rellena a cuadrado (Mira recorta el centro y perdía los lados de la calle) y Mira recibe nuestra profundidad, así sus mallas y nuestros planos comparten marco y escala. La colocación de Mira con gravedad dejó casas fuera de su máscara (IoU 0,0-0,3 en basque_street); `flow_refine.py` parte de la mejor pose de Mira y ajusta cada malla a los puntos de MoGe de su máscara (ICP con escala): de 0,008 a 0,65 en una casa, mejora en 6 de 8. `flow_check_meshes.py`: malla girada junto a su recorte, 3 preguntas votadas. Una instancia que SAM 3D no cabe en la GPU ni con el proceso limpio se suelta y el caso se rehace.

Fase 2 (Imitación):
- `piece_build.py` sin objetos (el lugar vacío) y suelo hasta todos los píxeles de suelo; textura del suelo solo de la mitad cercana, sin sombras y con proporción máxima 3:1 (un recorte en tira dibujaba rayas en toda la calle).
- `env_add.py`: zócalo hasta el suelo para la malla que flota (SAM 3D no hace las plantas tapadas por setos; bajarla la sacaba de su sitio), y fachadas planas de respaldo solo donde ninguna malla las cubre.
- `flow_gates.py` (pyrender desde la cámara de la foto): suelo cubre el 90 % de los píxeles de suelo hasta 60 m, error de profundidad, y por malla: base (flotar falla, hundirse no se ve), IoU de silueta >= 0,45 y profundidad. Atrapó el suelo de pocos metros con la casa flotando sobre el HDRI.
- `flow_compare.py`: render contra foto en los dos órdenes. No es fiable: facade_car suspendió en una pasada y aprobó en la siguiente sin mejorar, y la primera versión aprobó la casa flotante. Bloquean solo cielo, horizonte, suelo y flotar; "edificios iguales" y "bloque sobre la calle" se registran. Falta calibrarla con hojas etiquetadas.

Fase 3 (Creación de entorno): `crossroad.py` (ya existía) y `street_template.py` nuevo (eje de la calle por PCA de la calzada; lado no visto = copia girada alrededor de la vertical de la cámara; repetición a lo largo con otro ladrillo; fachadas planas repetidas solo si van a lo largo de la calle). `flow_check_views.py`: preguntas sí/no por vista; pasa con >= 60 % de vistas.

Resultado en las 7 fotos de día: corner_timber (cruce, 15/17 vistas), terrace_row (cruce, 13/17), facade_car (calle, 12/16, réplica floja), basque_street (calle, en curso), church_corner (pasa la fase 2, falta la plantilla de plaza), straight_whitewash (rechazada: fachadas vistas muy de lado, ni SAM 3D ni el plano de respaldo las hacen bien), straight_road (en curso). Revisión del usuario: bien en general pero demasiado claro, sobre todo el suelo; `piece_isaac.py` baja el albedo de lo recortado de la foto (x0,8, suelo x0,7; la foto ya trae su luz) y la exposición (luces altas a 0,85, mediana 0,28-0,52).

Coste del día en VLM: ~3 M de tokens, dos tercios en la revisión de máscaras (582 llamadas por los votos).

### Medidas deterministas antes de cerrar la fase 2: luz y fotometría (2026-10-07, idea del usuario)

- 2.3b `flow_light.py` + `piece_isaac.py --sweep`: la vista 0 con 25 luces fijas (sol en 8 direcciones a 20, 45 y 70 grados, y solo cielo); se queda la que menos difiere de la foto en una rejilla de píxeles, tras llevar cada imagen a la luminancia media de la foto (se compara dónde cae la luz, no cuánta). Es el cambio que más rinde: delta E medio de la vista 0 de 40 a 27,4 (basque_street), 39 a 25,6 (church_corner), 31 a 22,9 (straight_whitewash), 30 a 26,3 (corner_timber). El sol al azar iluminaba la casa principal por detrás y quemaba el suelo.
- 2.4a `flow_photometry.py`: 1 de cada 3 píxeles en cada eje (11 %), sin cielo ni lo borrado por LaMa. Global como el balance de blancos de una cámara (mundo gris): exposición = ratio de luminancias, tinte = ganancia por canal. Por objeto (mapa de ids de pyrender): delta E mediano, ratio de luminancia, ganancia y parte del error total, para nombrar qué ajustar; mapa de calor 8 x 6.
- Corrección del tinte global: funciona (corner_timber de 15 % a 4 % hacia el azul) pero el delta E casi no cambia (26,32 a 26,27): el error que queda es de geometría y textura.
- Ganancias por objeto: empeoraron 2 de 2 lugares (26,3 a 28,3) y el flujo vuelve atrás solo. Una malla no coincide píxel a píxel con la foto y una casa mezcla tejado, paredes y flores; además la exposición automática compensaba cada cambio (ahora la vista 0 se expone fija mientras se calibra). Queda como opción (`--mode object`), no por defecto. Un objeto 3 veces más oscuro o claro que la foto es luz, no color: no se toca.
- Exposición: en las fotos nubladas el render sale más claro (ratio 0,57-0,66 en facade_car, straight_road, church_corner). El ratio pasa al lugar (`exposure_scale`, 0,5-1,5) y la fase 3 lo aplica a todas las vistas: straight_road con mediana 0,42 en vez de 0,52. Es lo que el usuario vio a ojo (demasiado claro, sobre todo el suelo).
- La revisión por vista con VLM también es ruidosa: terrace_row pasó del 76 % al 59 % de vistas aceptadas solo con otra luz.

### Fotometría por proporciones entre objetos (2026-10-07, idea del usuario)

Primero se mide la relación foto / render de cada objeto y se compara con la de los demás: si todo el lugar solo estuviera más oscuro o teñido, todos compartirían el mismo ratio. El que se aparta más de un 10 % (en algún canal; primero se probó 25 %) está desproporcionado y se corrige a la referencia; después se aplica el mundo gris sobre el render ya corregido. Más de 3 veces de diferencia se marca como luz (fachada en sombra), no como color, y no se toca (`flow_photometry.py --mode relative`, ahora por defecto).

- Referencia = mediana pesada de todos los objetos: falla. El objeto más grande manda; en church_corner era la iglesia, cuya textura de SAM 3D salió blanca en vez de piedra, y la casa de la derecha, bien de color, se volvió rosa (ΔE 25,6 a 25,9).
- Referencia = suelo y fachadas planas (`--reference pieces`, por defecto): su textura sale de la propia foto, así que su ratio es solo luz y exposición. church_corner 25,6 a 23,7 (iglesia más cálida, cruz de piedra 27 a 15); basque_street 27,4 a 27,7, corner_timber 26,3 a 26,4, straight_road igual. En esos dos se deja el estado anterior (solo balance de blancos).
- Límites: una ganancia por malla no arregla materiales mezclados (tejado, pared y flores); el tope de ganancia 2 no alcanza a las casas en sombra de corner_timber, que son luz. Hojas: `data/street_test/flow/<foto>/photometry_global_vs_rel.jpg`.
- Umbral 10 % frente a 25 % (usuario): mismo resultado (basque 27,73 frente a 27,68, church 23,73 en ambos, corner_timber 26,42, straight_road 24,06). Casi todas las desviaciones ya pasaban del 25 %; se deja 10 % por defecto.

### photo2scene sin Mira-Scene y con la foto como textura (2026-10-07)

El flujo se llama photo2scene (usuario) y ya no llama a nada externo: `photo2scene.py` (antes `scene_flow.py`). Mira-Scene, sus entornos y sus pesos se borraron de Crucial (27 GB, usuario).

- Profundidad: MoGe-2 corre en `eyeforge-sam3d` con su checkpoint en `tools/moge_ckpt` y sus versiones de utils3d en `tools/moge_deps` (delante en `sys.path`, el entorno conserva las que pide SAM 3D). Misma salida bit a bit que en `mira-geometry`.
- Mallas: `flow_sam3d.py` lanza SAM 3D Objects entero (etapas 1 y 2 más la cabeza de pose) con nuestra profundidad como point map; la pose sale ya en nuestro marco (PyTorch3D a OpenGL). Unos 20 s por instancia (Mira, unos 2 min); IoU con la máscara 0,43-0,68 de salida y 0,45-0,93 tras `flow_refine.py`. Geometría más completa (inventa lo oculto) y el VLM acepta 27 mallas frente a 21; texturas más lisas que con el CCM de Mira.
- Textura de la foto (`env_add.py --photo-texture`): cada vértice se proyecta en `background.png` y guarda esa posición como coordenada de textura; las caras vistas (delante de las demás mallas, dentro de su máscara erosionada 2 px, no rasantes) usan la foto a resolución completa, el resto los colores de SAM 3D ajustados al tono de la parte vista. Antes se subdividen las caras orientadas a la cámara hasta 16 px en la foto (la decimación deja una fachada en pocos triángulos grandes y un vértice fuera de la máscara perdía el triángulo entero: 6 % de caras con foto en la casa de basque_street). En Isaac, un GeomSubset con material de textura. Las copias de las plantillas no llevan la foto (se rebrickean).

| Foto | ΔE Mira | ΔE SAM 3D | ΔE SAM 3D + foto | Vistas aceptadas (Mira, SAM 3D, + foto) |
|---|---|---|---|---|
| church_corner | 25,6 | 25,1 | 17,2 | (sin plantilla de plaza) |
| basque_street | 27,4 | 29,8 | 25,4 | 81 %, 100 %, 94 % |
| corner_timber | 26,3 | 29,5 | 21,8 | 94 %, 82 %, 76 % |
| facade_car | 34,4 | 30,0 | 14,7 | 75 %, 75 %, 94 % |
| straight_road | 24,1 | 24,2 | 15,5 | 88 %, 81 %, 88 % |
| straight_whitewash | 22,9 | 26,5 | 24,4 | 44 %, 44 %, 75 % |
| terrace_row | 22,7 | 26,6 | 19,9 | 59 %, 59 %, 77 % |

Seis de siete lugares quedan listos (church_corner espera la plaza). Exposición de la vista 0: se fijaba con su primer render y, con una luz que deja en sombra la fachada que la foto muestra al sol, terrace_row salía 4,5 veces más oscura que la foto (ΔE 35). Ahora, si la exposición final sale fuera de 0,8-1,25, se corrige la de la vista 0 y se mide otra vez (terrace_row 35 a 19,9, straight_road 23,7 a 15,5). Hoja: `data/street_test/flow/photo_texture_compare.jpg`. La revisión de vistas del VLM sigue siendo ruidosa (cambia sin que cambien las vistas). Pendiente: `run_photo.sh` y `run_hope_train.sh` (ruta de producto) aún llaman a Mira.

### CCM de Mira dentro de photo2scene (2026-10-07)

Solo el código de Mira-CCM (`tools/mira_src`, 69 MB) y sus pesos (`tools/ccm_ckpt/pipeline`, 6 GB), en `eyeforge-sam3d` con diffusers 0.35, transformers 4.48 y peft 0.17 aparte en `tools/ccm_deps` (solo los carga `flow_ccm.py`). Unos 19 s por instancia. `flow_sam3d.py --voxels sam3d|ccm|hybrid` y `photo2scene.py --voxels`; por defecto sigue `sam3d`. Prueba en basque_street (copias en `data/street_test/flow_ccm/`, hoja `sheet/compare.jpg`):

- `ccm`: la geometría sigue la foto (balcón corrido, ménsulas, puerta de la casa 0) y la malla sale cerrada, porque el generador de forma del CCM también inventa la trasera. SAM 3D inventa una casa genérica.
- La pose del CCM contra la profundidad de MoGe es floja (inliers 11-58 %): un ajuste afín encaja 2-3 veces mejor que una similitud, las fachadas casi planas dejan el grosor mal definido. Tras `flow_refine.py`, IoU con la máscara mejor en 4 de 9 y peor en 5.
- El lugar completo se para en 2.2b: las puertas de silueta tiran 4 de 7 mallas (casa 0: IoU 0,41 frente a 0,61 con SAM 3D) y otra tras rehacer.
- `hybrid` (vóxeles del CCM más los de SAM 3D detrás de lo visto, llevados al marco del CCM con su pose): las dos formas no casan y sale un amasijo; la casa grande no cabe en la GPU. Descartado.
- `flow_refine.py --axis-scale` (también `photo2scene.py --axis-scale`): tras la similitud, escala por eje de la caja orientada de la malla (ICP) y búsqueda directa sobre el IoU (escalas por eje y desplazamiento en x, y, paso que se parte a la mitad). CCM en basque_street: IoU 0,59-0,88 en las 9 mallas (antes 0,31-0,78) y pasan las 7 que llegan a las puertas de silueta. Pero el VLM tira la casa de los balcones ("foreign"): el CCM copia también lo que su máscara arrastra (portal y seto), mientras SAM 3D lo ignora. ΔE 21,8 (sin esa casa, no comparable) y el lugar se para en 2.4 por "floating", que es la casa del fondo sobre un bloque que también tiene la versión SAM 3D (que pasó).
- Mismo ajuste con SAM 3D: puertas de silueta 9 de 9 (antes 7), ΔE 26,3 (antes 25,4), vistas 81 % (antes 94 %). Neutro.

### photo2scene en interior: home_office (2026-10-08)

Foto de ejemplo de Mira (`data/scene3d_test/images/003_home_office.png`), ruta indoor_floor, plantilla room. Antes se paraba en la fase 2; ahora sale "place ready" (`data/scene3d_test/flow/home_office/extended`). Cambios:

- `bg_prepare.py --keep-inside <frases de entorno>` (photo2scene pasa las del VLM): un objeto móvil que está en más de un 70 % dentro de un mueble (máscara de SAM 3 con los agujeros rellenos, porque SAM 3 deja los libros fuera de la estantería) y no toca el suelo es su contenido y no se borra. Antes LaMa vaciaba la estantería y SAM 3 ya no la encontraba; ahora se quedan 12 de 30 instancias (libros, plantas y cestas en baldas) y pasan estantería, mesa y aparador. La mesa ya no la tira el VLM.
- `env_add.py`: un hueco bajo una malla menor de 10 cm se cierra bajándola (antes ni falda ni nada, y el filtro tira desde un 6 % de la altura; aparador a 7 cm).
- `photo2scene.py`: `piece_build.py --indoor` fuera de la ruta de calle (habitación cerrada con techo; antes siempre `--outdoor` y salía un cielo de campo) y `--drop-covered-walls` solo en calle (los muebles pegados a la pared la quitaban).
- `flow_compare.py` y `flow_check_views.py --indoor`: preguntas de habitación (paredes y techo, suelo, muebles flotando o ausentes) en vez de cielo y calle.
- `room_template.py` (fase 3, plantilla room): la habitación ya está cerrada, así que solo añade cámaras dentro, a 0,4-1,8 m, a 0,5 m de paredes y muebles, mirando a un mueble (4 de cada 5) o al centro de una pared. Mirando por igual a muebles y paredes, la mitad de las vistas eran pared lisa y el VLM las aceptaba.

Resultado: filtros geométricos con las 3 mallas (IoU 0,59, 0,71 y 0,82), ΔE 21,4, comparación del VLM bien en todo y 15 de 16 vistas aceptadas. Defectos a la vista: la ventana se pierde (la pared es una textura repetida de un trozo de la foto, con manchas), la estantería sale vacía y oscura por dentro y con parches de la foto borrada, y la cara vista del aparador lleva el relleno de LaMa.

Control en calle (basque_street desde 1.3a con estos cambios): solo una señal pegada a una casa deja de borrarse; 7 mallas pasan los filtros igual que antes, ΔE 25,4 a 25,3, comparación bien; vistas aceptadas 94 % a 75 % (3 rechazos por "street_view", revisión de vistas ya ruidosa antes).

### Borrado de objetos: LaMa frente a Qwen-Image-Edit (2026-10-08)

`erase_qwen.py`: Qwen-Image-Edit-2511 con el LoRA Object-Remover (ambos Apache 2.0), los objetos de `prepare.json` nombrados en el prompt, pegado solo dentro de la máscara de borrado de LaMa con borde suave. En la 3090 en 4 bits (nf4) con descarga a CPU, 40 pasos: unos 4,6 min por foto (LaMa, segundos). Cuantizar en 4 bits la modulación del primer bloque y las proyecciones de entrada y salida llena toda la imagen de ruido fino; se dejan en bf16. Entorno `/media/lenena-iker/Crucial/envs/eyeforge-qwenedit`, pesos en `/media/lenena-iker/Crucial/hf_cache` (54 GB). Hojas: `data/scene3d_test/erase_compare/` (original con la máscara / LaMa / Qwen).

- Qwen mucho mejor en 6 de 8: reconstruye lo que hay detrás (estantería y libros tras la silla en home_office, verja, zócalo y bordillo tras el coche en facade_car, fachada y escaleras en corner_timber, casas del fondo en terrace_row y church_corner). LaMa deja borrones y cuadros repetidos.
- Conserva las sombras que no son del objeto (sombra de la casa en corner_timber y church_corner), que LaMa borraba con la máscara de sombras; mejor para ajustar la luz.
- Fallos: no siempre quita lo que se le nombra (straight_road: deja una farola y una señal; el rótulo pintado en la fachada también se queda, que ahí es lo correcto). En straight_whitewash, igual que LaMa.

### Qwen en photo2scene y luz de interior (2026-10-08)

- `photo2scene.py` borra con Qwen por defecto (paso `1.3a_erase`, `--eraser lama` para volver). Se usa la imagen entera de Qwen, que también quita las sombras que la máscara de LaMa dejaba fuera (usuario: mejor la cruda), salvo que la correlación de fase fuera de la máscara la vea movida 0,5 px o más (basque_street, 1,4 px); entonces se pega dentro de la máscara. El resto de fotos: menos de 0,1 px.
- home_office con Qwen y la luz anterior salía peor que con LaMa (ΔE 25,2 frente a 21,4): el cuello de botella era la luz. El primer render era unas 4 veces más oscuro y cálido que la foto (panel de 1,2 m y lámpara), el barrido probaba soles apagados en interior (9 renders iguales) y la fotometría, con canales libres, teñía las paredes de verde y el aparador de azul.
- Cambios: panel en el 80 % del techo; ventanas de la foto (`piece_surfaces.py` busca "window", `piece_build.py` las proyecta sobre el plano de su pared con un anillo del 3 % de la foto, porque el marco las separa de la máscara de pared) con un panel de luz de día hacia dentro; barrido de interior por intensidades de panel, ventana y lámpara; `flow_photometry.py --max-tint 0.15` (cada canal a un 15 % de la ganancia de luminancia como mucho).
- home_office: ΔE 25,2 a 17,2 sin ventana y 14,8 con ventana, color neutro, 15 de 16 vistas. basque_street igual (25,3): en calle las ganancias ya estaban dentro del 15 %.
- Pendiente en interior: la ventana no se ve (las cuatro paredes comparten la textura de la pared principal), la estantería sale vacía, cámaras pegadas a los muebles. Solo validado en una foto de interior: hacen falta más.

### Escenario o borrar por tamaño medido, en interior (2026-10-08)

Foto nueva del usuario (`data/scene3d_test/flow/user_desk`, buhardilla con escritorio, gran angular del móvil). Con el flujo anterior salió "place ready" con la mesa grande perdida (silueta 0,30 tapada por la silla y cables), la estantería sin encontrar (SAM 3 con "shelf" solo vio trozos) y Qwen dejando monitores, lámpara y cajas. Además se paraba en el horizonte: las tres muestras del VLM daban 0,5, 0,42 y -0,1 (tomaba la unión pared-techo por el horizonte) y la geometría 0,26, que era la buena. `flow_camera.py` ya no usa el horizonte del VLM si sus muestras discrepan más de 0,3 (en las 10 fotos anteriores, como mucho 0,1).

Decisión con el usuario: se borran las clases objetivo y todo lo pequeño y movible (los distractores vuelven aleatorios desde el generador, para que el modelo no memorice dónde están); el escenario es lo grande. "Segmentar todo" con SAM 3 por rejilla de puntos (`segment_all.py`, 13 s) da sobre todo partes (cajones, láminas del radiador) y nunca la mesa o la cama enteras, así que la escala la elige el VLM:

- `flow_things.py`: el VLM lista cada cosa entera con su caja ("una mesa con sus patas, no su tablero") y SAM 3 (`Sam3TrackerModel`, prompt de caja) da la máscara. 30 cosas en user_desk, unos 30 s.
- `bg_depth.py --only photo` y `scene_split.py`: con los puntos métricos de MoGe y el plano del suelo, alto y tamaño de cada cosa. Lado mayor de 0,6 m o más: escenario con malla; pequeño en pared (enchufe, interruptor): textura; resto y clases objetivo (`--targets`): borrar. Medidas creíbles (mesa 0,77 m de alto y 1,25 de ancho, cama 0,62, monitores 0,5); se quedan cortas en lo tapado o visto de canto (radiador 0,41 de ancho, pasa por el alto).
- `bg_prepare.py --object-masks` borra esas máscaras; `flow_segment.py --from-split` pide a SAM 3 las cajas del escenario sobre la foto ya borrada (la mesa incluye lo que tapaba la silla).
- Solo en rutas de interior. En calle la regla de tamaño borraría mal (un coche es grande y es el objetivo): sigue como antes hasta que photo2scene reciba las clases objetivo.

Resultado: user_desk pasa con las 5 piezas (cama, dos mesas, estantería, radiador), el VLM responde que los muebles coinciden y que no falta nada; Qwen limpia la mesa pero se deja un monitor, la lámpara y cajas de la estantería (24 nombres en el prompt). home_office: estantería, mesa, aparador, silla y lámpara de pie como mallas, Qwen vacía la estantería limpia, ΔE de la luz 12,1 a 10,9, comparación bien, 15 de 16 vistas. Pendiente: lo que Qwen deja sin borrar (comprobar con SAM 3 y repetir), la mesa del primer plano de user_desk sale con parches grises, umbral de 0,6 m fijo hasta tener el tamaño de la clase objetivo.

### Ruta barata "support" en photo2scene (2026-10-08)

Pregunta del usuario: ¿distingue el pipeline una escena simple (botellas en una mesa, una cámara fija sobre una cinta) de un interior o un exterior? Solo a medias: había ruta `table`, pero entraba por el camino de habitación y paraba antes de la fase 3; la cinta o una foto al suelo acababan en out_of_scope. Cambios:

- `flow_observe.py`: lugares nuevos `work_surface` (mesa, escritorio, balda, cinta, banco de trabajo) y `ground_close` (suelo, césped o tierra vistos de cerca o desde arriba); la ruta `table` pasa a llamarse `support`. `flow_route.py`: esos dos lugares van a `support` dentro o fuera y con cualquier luz (solo cuenta que el VLM dude del lugar); `--force support` para cuando el spec ya diga que la cámara es fija. Las 10 fotos anteriores mantienen su ruta con el prompt nuevo.
- `photo2scene.py`, pasos S.1 a S.7: el antiguo camino de HOPE (`bg_plan`, `bg_prepare`, `bg_build`, `bg_isaac`, `view_labels`, `bg_review`) sin Mira: `flow_sam3d.py --masks instances.npz --image photo --out meshes_objects` hace las mallas de los objetos de la foto y guarda la malla canónica con su rotación a la cámara, que `bg_build.py` usa como orientación inicial (antes la daba el CCM de Mira). `run_photo.sh` queda superado.
- `erase_qwen.py`: Qwen no siempre borra lo que se le nombra (botellas: de 3 botellas y 3 tapones quitó una botella). Un objeto cuya máscara cambia menos de 12 de media (0-255) sigue ahí: medido en 5 fotos, los que quedan cambian 4-10 y los borrados 15 o más. Segunda pasada nombrando solo esos y, lo que siga, relleno de LaMa en su máscara. Botellas: tras la segunda pasada quedaban 2 tapones, LaMa los quitó (deja dos círculos tenues).
- `bg_review.py`: la hoja estiraba las fotos verticales a 16:9 y el VLM veía formas estiradas y tapas enormes que no lo eran; ahora las casillas mantienen la forma de la foto.

Resultado: botellas, 15 min (Qwen 2 pasadas 8,6 min, SAM 3D 2,3 min, Isaac 2,3 min), revisión 3 (se queda): falla la cafetera, que SAM 3 no encontró con "moka pot" y queda en el fondo sin etiqueta. Baldosas con hojas mirando al suelo (`data/real/grass`): ruta support sola, 16 hojas borradas y recolocadas, revisión 3: sobran cajas diminutas de restos que SAM 3 tomó por hojas. HOPE y una cinta transportadora sin probar.

### Borrado más rápido: Qwen Lightning frente a otros (2026-10-08)

Qwen-Image-Edit a 40 pasos tarda 260 s por pasada en la 3090. Sin benchmark público que compare los candidatos con Qwen-Image-Edit, se probaron en las fotos ya medidas (`erase_bench.py`, `data/scene3d_test/flow/<foto>/erase_bench/`): mismas cosas a borrar, tiempo, objetos que quedan (cambio medio menor de 12 en su máscara) y cambio fuera de la máscara.

- user_desk (25 cosas): Qwen 40 pasos 260 s, quedan 10; Qwen con el LoRA Lightning de 4 pasos (lightx2v, Apache 2.0) 23 s, quedan 12, imagen casi igual; FLUX.2 klein 4B (Apache 2.0) 20 s, quedan 7 pero deja monitores y silla y cambia el resto de la foto (6,8 fuera de la máscara frente a 2,4); RORem (SDXL con máscara) 2,3 s, borra a base de emborronar la mesa entera. LaMa y RORem descartados por el usuario.
- Por pasada Lightning deja más (home_office 12 de 24 frente a 1), pero 3 pasadas nombrando lo que queda salen más limpias que una de 40 pasos en user_desk, home_office y botellas, en 45-90 s. Queda por defecto en `erase_qwen.py` (`--full` para 40 pasos). En las botellas también quitó la cafetera, que no se le nombró.
- Descartados por licencia: ObjectClear y TurboClear (los mejores en los benchmarks de borrado, 1,7 s y 0,04 s, pero pesos con licencia NTU S-Lab no comercial).
- LaMa sigue solo como último recurso para lo que queda tras las 3 pasadas (1 objeto en 3 fotos).

### Lo borrado como assets para el generador (2026-10-08)

Decisión del usuario: todo lo que se borra (objetivos y distractores) debe salir de la fase 1 como asset, para recolocarlo al generar. `flow_assets.py` (paso 1.4b en calle y habitación, justo tras la profundidad para que salgan aunque pare una puerta posterior; S.4 en la ruta support) junta lo pedido (instances.npz) y lo que el editor quitó sin pedírselo: zonas fuera de las máscaras con cambio medio difuminado mayor de 30 y más del 0,2 % de la foto; el VLM ve cada zona antes y después y nombra lo que desapareció o dice que no hay objeto; SAM 3 la recorta con su caja. Tamaño con la profundidad métrica y el plano de apoyo; papel `target` si el nombre está en `--targets`, si no `distractor`. Luego `flow_sam3d.py --masks assets.npz --out assets` hace las mallas canónicas. Catálogo en `<dir>/assets.json`; en la ruta support `bg_build.py` usa esas mismas mallas.

- Botellas con Lightning: 8 assets (3 botellas, 3 tapones y las dos piezas de la cafetera, que Qwen quitó sin pedírselo y el VLM llamó "metal container"; una tercera zona descartada). Mallas buenas. Ruta support completa en unos 9 min (antes 15) y revisión 5 (antes 3).
- user_desk: 25 assets en 7,6 min de SAM 3D (unos 18 s cada uno). Bien: lámpara, monitores, teclado, ratón, silla, gafas y cajas. Flojos: papeles y libretas como losas planas, y una libreta con las gafas dentro (máscaras solapadas).

### Lo borrado vuelve al azar en cada vista (paso 3.5, 2026-10-08)

`flow_populate.py` coloca en cada vista aceptada del lugar extendido (habitación o calle) entre 2 y 8 assets de la fase 1, con un sorteo nuevo por vista (qué assets, dónde y con qué giro), para que el modelo no aprenda dónde está cada distractor. Superficies: mapa de alturas visto desde arriba (pyrender ortográfico, sin techo); una huella vale si todas las celdas bajo ella están a la misma altura (±1 cm), así el objeto queda apoyado entero, sin atravesar paredes ni muebles, y puede ir en el suelo, en una mesa, en una balda o encima de otro asset. Nunca bajo una balda (el mapa solo ve lo de arriba). Solo puntos que la cámara ve (prueba de profundidad), a más de 0,5 m y con al menos 24 px de tamaño. Tamaño real: malla canónica de SAM 3D por la escala de su pose (comprobado: coincide con la malla posada en métrico; portátil 0,29 m, teclado 0,42 m). Luego `piece_isaac.py` renderiza y `view_labels.py` etiqueta: con `roles` en scene.json, cada caja lleva nombre y papel (target o distractor) y los muebles del lugar no llevan caja.

Ojo: pyrender linealiza la profundidad de cualquier cámara como si fuera de perspectiva; con la ortográfica hay que deshacerlo (el mapa salía todo a la altura del techo).

user_desk: 25 assets, de 2 a 8 por vista en 17 vistas, todos usados; cajas alineadas con el render de Isaac. Tiempos en la 3090: colocar 101 s (casi todo cargar y simplificar las 25 mallas de SAM 3D), render 19 s, etiquetas 3 s. Pendiente: más peso a mesas y baldas (ahora casi todo cae al suelo por área), algún monitor sale tumbado (eje vertical canónico de SAM 3D no siempre es el real), prueba A/B con y sin distractores.

Calle (facade_car): coche y moto en sitios distintos por vista, apoyados en la calzada o la acera. `view_labels.py` usa los cuadriláteros del lugar para las oclusiones cuando no hay malla de fondo. El detector de cambios de `flow_assets.py` había sacado dos "blue car" planos (bordes y sombra del coche fuera de su máscara) que salían como losas, una encima del coche: ahora una zona que toca una instancia pedida y comparte palabra con su nombre se descarta como trozo de ella. Queda un "pipe" (un poste de la valla) como asset raro pero válido.
