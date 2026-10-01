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

## Regresión

Una cabeza con 1 a N salidas continuas y pérdida de error cuadrático o absoluto. El profesor lo tiene difícil (un VLM estima mal ángulos y distancias), así que aquí el valor lo pone Isaac: cada render sabe el ángulo, la distancia o el nivel exactos. Las fotos del usuario sirven de test, marcadas a mano (por ejemplo, arrastrando una línea en Review).
