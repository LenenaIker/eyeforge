# Tipos de modelo que ofrece EyeForge

Decisión del usuario (2026-09-28): ofrecer ahora solo las tareas que caben en un ESP32 y cuestan poco de añadir; las de esfuerzo medio, más adelante si se ve bien. El resto queda como hoja de ruta.

## Qué se comparte y qué cambia por tarea

Se comparte: backbone (MobileNetV2 dimensionado por el perfil de hardware), cuantización int8 y el flujo (entrevista, datos del usuario, profesor, sintético, entrenamiento, export).

Cambia por tarea: la cabeza y la pérdida del modelo, el formato de las etiquetas (clase, puntos, cajas, máscaras, valores), lo que se pide al profesor, lo que se saca de Isaac, la métrica y el postproceso del firmware.

Isaac Replicator da gratis y perfectas cajas, máscaras, profundidad, puntos clave y pose de cada render: cuanto más cara es la etiqueta a mano (máscaras, distancias), más aporta el sintético.

## Tabla de tareas

| Tarea | Ejemplo casero | Qué cambia en el modelo | Profesor | Isaac | ESP32 | Pi / Jetson | Esfuerzo | Estado |
|---|---|---|---|---|---|---|---|---|
| Clasificación (imagen o zona fija) | ¿Hay bici en el cruce? | cabeza de clases | GLM, SAM 3 | sí | sí | sí | hecho | **ofrecida** |
| Conteo por puntos (rejilla FOMO) | Contar macetas o gallinas | rejilla con centros | SAM 3 | sí | sí | sí | hecho | **ofrecida** |
| Rejilla por zonas | Cortacésped: césped, no césped, borde | ninguno: clasificador por trozo (ver abajo) | SAM 3 con máscaras | máscaras | sí | sí | bajo | **ofrecer ahora** |
| Regresión | Ángulo para seguir una línea o pasillo, nivel de llenado | cabeza de 1 a N valores | difícil para un VLM | exacto | sí | sí | bajo | **ofrecer ahora** |
| Distancia a obstáculos por columnas | Robot aspirador: espacio libre delante | regresión por columna | SAM 3 + profundidad | profundidad exacta | sí | sí | medio | más adelante |
| Segmentación semántica ligera | Suelo transitable, línea en el suelo | codificador + decodificador pequeño (máscara 24x24 en ESP32) | SAM 3 con máscaras | máscaras | justo (S3) | sí | medio | más adelante |
| Eventos de vídeo con fotogramas apilados | Entró o salió, dirección del cruce | entrada de 2 o 3 fotogramas o su diferencia | VLM con varios fotogramas | secuencias | sí, baja resolución | sí | medio | más adelante |
| Detección con cajas | Localizar la base de carga, pelotas | cabeza SSD o YOLO-nano | SAM 3, LocateAnything | cajas | muy justo | sí | medio | más adelante (Pi o Jetson) |
| Anomalía | "Algo raro" en una estantería o cinta | solo fotos normales, sin etiquetas | no hace falta | poco | no | sí | medio | hoja de ruta |
| Puntos clave | Asa de un objeto, orientación de la base | mapas de calor | LocateAnything (apunta) | exacto | no | sí | alto | hoja de ruta |
| Segmentación de instancias | Separar cada fruta para recogerla | Mask R-CNN o YOLO-seg | SAM 3 | sí | no | Jetson | alto | hoja de ruta |
| Modelos temporales (3D, recurrentes) | Reconocer acciones | red temporal | VLM de vídeo | secuencias | no | Jetson o GPU | alto | fuera de alcance |

## Vídeo sin tarea nueva

La mayoría de robots no necesita un modelo de vídeo: el modelo de siempre por fotograma, más votación temporal (avisar si sale en k de n fotogramas), disparo por movimiento (diferencia de fotogramas) o seguimiento en el firmware. No cambia el entrenamiento; va en el firmware de ejemplo.

## Rejilla por zonas: clasificador por trozo

No es un tipo de modelo nuevo. Primera versión (A): la imagen se parte en N trozos (6 como GreatScott!), cada trozo se clasifica con el clasificador de siempre y el firmware recorre los trozos.

| | A. Recortar y clasificar cada trozo | B. Un modelo con salida por celda |
|---|---|---|
| Modelo | el clasificador actual | cabeza nueva (como la del conteo) |
| Inferencias en el chip | N por imagen | 1 |
| Resolución por celda | más alta (cada trozo se amplía a la entrada) | más baja |
| Contexto | cada trozo no ve a sus vecinos | ve la imagen entera |
| Esfuerzo | muy bajo | medio |

B queda como mejora si hace falta velocidad o contexto. En los dos, el profesor etiqueta igual: la clase de cada celda sale de la máscara de SAM 3 (qué parte de la celda cubre cada clase), sin mirar celda a celda.

### Prueba A frente a B: caminos (2026-10-02)

Dataset del usuario `data/real/roads`: 74 fotos de móvil (calles, aceras, carriles bici, caminos), rejilla 3x3, dos clases: transitable (todo lo que se puede andar, rodar o conducir) o no. Etiquetas de SAM 3 sin revisar (`experiments/road_grid/sam3_masks.py`: unión de las máscaras de "road", "sidewalk", "bike lane", "footpath", "dirt path", "crosswalk", "parking lot"); una celda es transitable si al menos la mitad lo es. A simple vista las máscaras salen muy bien, con fallos pequeños (árboles en primer plano, una franja lejana). Comparación en `experiments/road_grid/compare.py`, 5 repartos por foto, int8, resultados en `runs/road_grid/cmp/results.json`.

B es la foto entera con 9 salidas (el mapa de rasgos reducido a 3x3 y aplanado, así cada salida ve posición y contexto). Cada fila es la mejor de dos recetas (solo la última capa o la red entera). Solo con la posición de la celda ya se acierta el 91% (arriba nunca hay camino, abajo en el 80% de las fotos), así que la columna que manda es el acierto en las celdas donde la posición falla (unas 60 de 666).

| Hardware | Variante | Acierto | Donde la posición falla | Fotos con las 9 bien | ms por foto (PC, 1 hilo) | Tamaño |
|---|---|---|---|---|---|---|
| | Solo la posición | 91,0% | 0% | 47% | | |
| ESP32 | A: recortes a 96 px | 91,6% | **57%** | 51% | 1,3 | 612 KB |
| ESP32 | B: foto a 96 px | 92,6% | 22% | 54% | 0,2 | 710 KB |
| ESP32 | B: foto a 288 px (mismo cómputo que A) | **94,0%** | 43% | 61% | 1,3 | 710 KB |
| Móvil/PC | A: recortes a 224 px | **96,6%** | **78%** | **72%** | 39,7 | 6,8 MB |
| Móvil/PC | B: foto a 224 px | 93,7% | 52% | 57% | 4,4 | 6,9 MB |
| Móvil/PC | B: foto a 672 px (mismo cómputo que A) | 92,3% | 37% | 54% | 40,2 | 6,9 MB |

- B con la misma entrada aprende sobre todo la posición: en la ESP32 acierta el 22% de las celdas donde hay que mirar, A el 57%.
- Con el mismo cómputo, en móvil y PC gana A con claridad (96,6 frente a 92,3%); más resolución no ayuda a B con tan pocas fotos. Causa probable: A entrena con 666 trozos y B con 74 fotos.
- En la ESP32 queda dividido: B a 288 px acierta más en total, A más donde hace falta mirar.
- El precio de A es la velocidad: 9 pasadas (unos 0,6 s por foto en el Pixel 7a con el clasificador del móvil, 64 ms cada una). En la ESP32 real sin medir.
- Cautela: etiquetas sin revisar y solo unas 60 celdas difíciles (cada una vale casi 2 puntos en esa columna); diferencias de 2 o 3 puntos son ruido.
- Decisión (usuario, 2026-10-02): ofrecer las dos según prefiera calidad (A) o velocidad (B).
- Pendiente para B: el aumento de datos de la prueba evita desplazamientos y giros porque mueven objetos de celda. Con las máscaras guardadas se pueden desplazar o recortar las fotos y recalcular la etiqueta de cada celda, lo que rompería la dependencia de la posición (camino también arriba). Sin probar.

## Regresión

Una cabeza con 1 a N salidas continuas y pérdida de error cuadrático o absoluto. El profesor lo tiene difícil (un VLM estima mal ángulos y distancias), así que aquí el valor lo pone Isaac: cada render sabe el ángulo, la distancia o el nivel exactos. Las fotos del usuario sirven de test, marcadas a mano (por ejemplo, arrastrando una línea en Review).
