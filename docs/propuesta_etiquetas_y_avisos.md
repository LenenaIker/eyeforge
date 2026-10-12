# Propuesta: clases comprensibles y avisos sobre los datos (tareas 5 y 9, 2026-10-12)

Propuesta para que decida el usuario: nada de esto está aplicado en el producto. Hoja de comparación con capturas,
mocks y los avisos de cada dataset: `/media/lenena-iker/Crucial/runs/eyeforge/p2s_test/compare/propuestas_etiquetas_avisos.html`.
Prototipo de los avisos: `experiments/data_warnings/checks.py` (y `sheet.py`, que genera la hoja).

## Parte A (tarea 5): clases y etiquetas para no expertos

### Cómo se ve hoy

Capturas del servidor local (`web_demo`, puerto 8782) en la hoja. Lo que cuesta entender:

1. La clase es un punto de color con su clave tal cual (`sun_cream`, `no_grass`, `water_bottle`) y "No description" cuando no hay descripción (`Labeling.tsx`, `ClassItem`).
2. Ninguna foto de ejemplo por clase: para saber qué es "border" hay que abrir fotos una a una.
3. Jerga: "Labeling", "marks", "teacher", "reviewed", "No marks", "Let the teacher mark them". Las cajas del profesor son discontinuas y las revisadas continuas, pero no hay leyenda.
4. El color sale del orden de la clase (`classHue` en `Dataset.tsx`, `hue` en `Review.tsx`) y no del campo `color` que ya guarda la tabla `class` de `library.db`: la botella sale roja en pantalla y es azul `#2f6db5` en la base.
5. El progreso es por etiquetado (barra de revisadas); por clase solo "103 marks", sin cuántas fotos la tienen ni cuántas revisó el usuario.
6. De paso: el perfil del dataset "contador mesa" dice "sunglasses" en un dataset de botellas (perfil hecho con 1 foto). No es de esta tarea, pero confunde en la misma pantalla.

### Qué hacen las herramientas y qué recomiendan

| Herramienta o guía | Qué hace con las clases | Fuente |
|---|---|---|
| Teachable Machine (Google) | Cada clase es un panel con sus ejemplos a la vista mientras se graban; nombres simples; evita "training" e "inference" para quien empieza; no impone una clase "neutral". | [Google Design, "Geek chic: teachable computers"](https://design.google/library/geek-chic-teachable-computers) |
| Apple Create ML | Una carpeta por clase; recomienda empezar con unas 30 imágenes anotadas por clase y equilibradas. | [WWDC19 424, notas](https://wwdcnotes.com/notes/wwdc19/424) |
| Lobe (Microsoft) | Pensado para gente sin experiencia en datos: importar fotos y etiquetarlas con un clic, una clase por nombre. | [Microsoft News, Lobe](https://news.microsoft.com/source/features/ai/lobe-machine-learning/) |
| Roboflow | Health Check: cuántas anotaciones por clase, clases infrarrepresentadas, imágenes sin anotar, mapa de dónde caen las cajas. Color por clase asignado solo (los usuarios piden poder elegirlo y ver el nombre sin pasar el ratón). | [Roboflow, Dataset Health Check](https://docs.roboflow.com/datasets/dataset-health-check), [Roboflow Annotate](https://docs.roboflow.com/annotate/use-roboflow-annotate), [foro: colores de clase](https://discuss.roboflow.com/t/can-i-change-classes-color/1559) |
| Label Studio | Cada etiqueta tiene color de fondo y atajo de teclado propios. | [Label Studio, Label tag](https://labelstud.io/tags/label) |
| CVAT | Cada etiqueta tiene nombre y color (automático si no se da). | [CVAT, labels and attributes](https://www.mintlify.com/cvat-ai/cvat/projects/labels-and-attributes) |
| Edge Impulse | Cola de etiquetado con ayuda de un modelo; solo enseña lo pendiente. | [Edge Impulse, labeling queue](https://docs.edgeimpulse.com/studio/projects/data-acquisition/labeling-queue) |
| Nielsen, heurística 2 | Hablar el idioma del usuario, sin términos del sistema. | [NN/g, 10 usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/) |
| Google PAIR Guidebook | Explicar al usuario qué aprende el sistema de sus datos y fijar expectativas (modelo mental). | [People + AI Guidebook](https://pair.withgoogle.com/guidebook) |

Lo común: un color fijo por clase igual en la lista y en las cajas, ejemplos visuales de cada clase, recuento por clase a la vista y vocabulario sin jerga. Nadie enseña la clave interna al usuario.

### Propuesta A: tarjetas de clase con fotos de ejemplo (1 a 1,5 días)

Cada clase es una tarjeta (estilo Teachable Machine o Create ML): nombre legible, su color como borde (el de la base, igual que sus cajas), 3 o 4 recortes de ejemplo, la descripción en palabras del usuario ("What counts as one") y una barra con "103 found, in 36 photos, 0 checked by you". Los avisos de la parte B de esa clase salen dentro de su tarjeta.

- Ejemplos: en contar, recortes de las cajas del profesor con más puntuación (mejor si son de fotos y objetos distintos; un paso más sería elegirlos variados con SigLIP 2); en clasificar, las fotos con más confianza o las revisadas.
- Coste: endpoint de recortes por clase con caché (servidor, unas 3 h), tarjeta y recuentos por clase (web, unas 4 h), usar `class.color` en todo (1 h), pruebas.
- Pro: es lo que más ayuda a entender una clase sin leer. Contra: un endpoint nuevo y más imágenes por página.

### Propuesta B: palabras llanas, leyenda y color fijo (medio día, solo web)

Sin cambiar la estructura de las páginas:

- Nombre visible derivado de la clave (`sun_cream` pasa a "Sun cream"); la clave sigue igual para SAM 3 y la base.
- Vocabulario: Labeling pasa a "What to find" (contar) o "How to sort" (clasificar); marks a "found" o "sorted"; teacher a "AI"; reviewed a "checked by you"; No marks a "Nothing found"; "Let the teacher mark them" a "Let the AI find them".
- Leyenda fija junto a la rejilla: discontinuo "the AI found it, not checked yet", continuo "you checked it".
- Color de `class.color` en lista, cajas y revisión (hoy cada página calcula el suyo por orden).
- Por clase: "103 found in 36 photos, 0 checked" en lugar de "103 marks".
- Coste: unas 4 h en `Dataset.tsx`, `Labeling.tsx`, `Review.tsx` y `library.tsx`. Riesgo bajo.

### Propuesta C: tablero por clase, como carpetas (2 días)

Una columna por clase con sus fotos (clasificar) o con los recortes de sus objetos (contar), estilo Lobe o carpetas de Create ML; arrastrar una foto a otra columna la reclasifica y la marca como revisada. El recuento y los avisos van en la cabecera de cada columna.

- Coste: arrastrar y soltar accesible y en móvil (toque largo), recortes por objeto para contar, guardar en lote. Lo más caro y lo que más cambia el flujo de revisión que ya funciona.

### Recomendación

B ya (barata, quita la jerga y el fallo de colores), A después si da tiempo antes de la entrega (la tarjeta es también el sitio natural de los avisos de la parte B), C después del hackathon.

## Parte B (tarea 9): avisos amarillos sobre los datos

### Dónde están las marcas

- `library.db` (fuente de verdad): tabla `mark` con centro `x, y` y tamaño `w, h` en 0..1 de la foto (`w, h` nulos en un clic del usuario), `source` teacher o user; `review` dice qué fotos revisó el usuario; `class` trae `key`, `description` y `color`.
- `input/labeling.json` del trabajo: copia fija del etiquetado (marcas del usuario y fotos revisadas).
- `label/verdicts.json` del trabajo: lo que usa el entrenamiento, marcas del profesor con las del usuario ganando en las fotos revisadas; en contar, cada objeto con `box`.
- Tamaño de entrada del alumno: `sim2chip/hardware.py`; contar usa `Hardware.count_input()` (ESP32: rejilla a 96 px; genérico: PicoDet a 320 px), clasificar `input_size` (96 o 224). La foto se aplasta a un cuadrado (`ingest`, `detector.load`), así que cada eje escala por separado.

Hoy ya existen dos avisos, solo en el registro y en el informe final: `counting.small_objects` (al menos el 50 % de los objetos bajo 8 px) y `counting.crowded` (al menos el 5 % de fotos con más de 30 objetos), calculados en `stages/label.py` y copiados al export.

### Comprobaciones propuestas y umbrales (decide el usuario)

| Aviso | Regla propuesta | Por qué ese umbral |
|---|---|---|
| Objetos pequeños | raíz del área de la caja a la entrada del alumno < 8 px en al menos el 25 % de los objetos de una clase | 8 px es la celda de nuestra rejilla (`counting.STRIDE`) y el paso más fino de PicoDet; FOMO de Edge Impulse reduce 1/8 por la misma razón ([FOMO](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices)); Vertex AI pide cajas de al menos 8 x 8 px ([Vertex AI, object detection data](https://docs.cloud.google.com/vertex-ai/docs/image-data/object-detection/prepare-data)); COCO llama pequeño a menos de 32 x 32 px y ahí la precisión cae a la mitad ([survey small objects](https://arxiv.org/pdf/2503.20516)). Medido en casa: macetas con mediana de 6 px a 224 dieron F1 9 % (2026-10-01). El 25 % (hoy 50 %) avisa antes de que sea la mayoría: con la mitad ya se pierde la mitad del recuento. |
| Objetos que el modelo fundiría | al menos el 10 % de los objetos de una clase. Rejilla (ESP32): centros en la misma celda o en una vecina (así une `counting.decode` los blobs). PicoDet: IoU > 0,6 con otro de su clase (`detector.NMS_IOU`) | Edge Impulse: con FOMO no se pueden separar objetos de centros solapados y recomienda que no estén muy juntos ([FOMO](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices)). Cada par fundido resta uno al recuento: con el 10 % el error ya se nota en la cifra. Variante más estricta en la hoja (`same_cell`: solo la misma celda). Tocarse sin fundirse no avisa: PicoDet solo descarta una caja si solapa más de 0,6 con otra. |
| Pocas fotos por clase | menos de 10 fotos: aviso fuerte; menos de 30: aviso suave | Vertex AI exige al menos 10 imágenes por etiqueta ([clasificación](https://docs.cloud.google.com/vertex-ai/docs/image-data/classification/prepare-data), [detección](https://docs.cloud.google.com/vertex-ai/docs/image-data/object-detection/prepare-data)); Apple recomienda empezar con 30 por clase ([WWDC19 424](https://wwdcnotes.com/notes/wwdc19/424)). En contar cuenta las fotos donde sale la clase. |
| Test pequeño | con 30 o más fotos, menos de 20 en el test de esa clase (hoy el test se lleva la mitad cuando hay pocas) | `ingest.MIN_TEST = 20`: por debajo, una foto mueve la nota 5 puntos o más. Solo si no salta ya "pocas fotos", para no repetir. |
| Desequilibrio | la clase mayor tiene 3 veces o más fotos (clasificar) u objetos (contar con varias clases) que la menor | Sin fuente directa para 3: Vertex AI tolera hasta 100 a 1 con miles de fotos; Roboflow y Create ML piden equilibrio sin cifra ([Health Check](https://docs.roboflow.com/datasets/dataset-health-check)). Con pocas fotos solo aprende la última capa (`train.HEAD_ONLY_BELOW = 100`) y la clase mayor arrastra las dudas. Criterio propio, a medir. |
| Opcional: cortados por el borde | al menos el 30 % de los objetos de una clase tocan el borde de la foto | Medio objeto se ve distinto de uno entero. Sin fuente, informativo. |
| Opcional: nota contra el profesor | menos de 20 fotos revisadas | Decisión A de `plan_etiquetado.md`: el test se completa con fotos del profesor y la nota sale optimista. |
| Ya existe: aglomeraciones | `counting.crowded` sin cambios | Medido en FSC-147 (2026-10-01). |

Clasificar no tiene tamaño de objeto: solo aplican pocas fotos, test pequeño y desequilibrio. La clase de fondo (`empty`, "none of the other classes") sin fotos no avisa.

### Qué saltaría hoy (salida de `checks.py`)

Con el hardware de cada trabajo y, como "y si", con ESP32-CAM. Texto exacto en la hoja.

| Dataset | Hardware | Avisos |
|---|---|---|
| bottles_classify (25 + 25) | genérico y ESP32 | pocas fotos suave en las dos clases (25 < 30) |
| bottles_few y bottles_scene (mismo etiquetado, 50 fotos, 103 botellas, mediana 71 px a 320 y 21 px a 96) | genérico y ESP32 | test pequeño (unas 18 fotos); opcionales: 30 % cortadas por el borde, 0 revisadas. Ninguna pequeña ni fundida. |
| grass (69 fotos: 41, 12, 16) | ESP32 | pocas fotos suave en border y no grass; desequilibrio grass 41 contra border 12 (3 a 1) |
| roads_cars (74 fotos, 207 coches, mediana 8,6 px a 320) | genérico (PicoDet 320) | **objetos pequeños: 46 % bajo 8 px** (el aviso actual del 50 % no salta); opcional 0 revisadas |
| roads_cars | ESP32 (y si) | objetos pequeños 86 % (mediana 2,6 px); fundidos 81 % (misma celda 54 %) |
| biblioteca web_local, Grass border no grass (28, 15, 26) | genérico y ESP32 | pocas fotos suave en las tres |
| biblioteca e2e_labeling, Apples (24 fotos, 871 manzanas) | genérico | pocas fotos suave (23 fotos con manzana); aglomeraciones (ya existe, 46 %) |
| Apples | ESP32 (y si) | además pequeños 43 % y fundidos 70 % |
| FSC-147 apples, geese, strawberries | genérico | solo aglomeraciones (ya existe) |
| FSC-147 en ESP32 (y si) | ESP32 | pequeños: manzanas 51 %, gansos 80 %, panecillos 27 %; fundidos 45 a 85 % |
| FSC-147 bread_rolls | genérico | ninguno |

Lectura: en el genérico casi todo pasa, salvo los coches de calle, donde el aviso propuesto salta y el actual no. En ESP32-CAM los conjuntos de contar con muchos objetos fallan casi todos: es justo lo que el aviso debe decir antes de entrenar, junto con qué hardware lo resolvería.

### Textos para el usuario (inglés llano)

Ejemplos reales del prototipo (plantillas en `checks.py`):

- Pequeños: "86% of your car marks are too small for this device. On ESP32-CAM the model sees each photo at 96 x 96 px, and those end up smaller than 8 px (half of all of them under 3 px): it will miss most of them. Take the photos closer, so each car fills more of the picture."
- Fundidos: "81% of your car marks are so close to another one that the model on ESP32-CAM will see two as one, and count fewer than there are. Photos where they stand apart, or taken closer, count better."
- Pocas fotos (fuerte; cifra de ejemplo, no salta en ningún dataset): "Only 6 photos are sorted as "border". The model needs at least 10 to learn anything about it, and 30 or more to do it well. Add photos of border in different places and light."
- Pocas fotos (suave): "12 photos are sorted as "border". That is enough to start, but 30 or more, in different places and light, make the model much steadier."
- Test pequeño: "About 18 photos of "bottle" are kept apart to test the model, so its score for this class can be off by 5 points or more. More photos make the score trustworthy."
- Desequilibrio: ""grass" has 41 photos and "border" only 12 (3 times fewer). The model will lean towards grass when unsure. Add photos of border, or leave some of grass out."

Mejora posible: cuando otro perfil lo resuelve, una línea más ("On a phone they would be 9 px instead of 3."), calculada igual.

### Dónde se enseñan

1. **Tras marcar el profesor la muestra de calibración y antes de entrenar** (el sitio principal): mensaje amarillo del sistema en el chat y lo mismo encima del panel de revisión del trabajo, con botones "Check the photos", "Show me which" (filtra las fotos afectadas) y "Train anyway". Nunca bloquea.
2. **Al elegir en el chat un etiquetado ya hecho** (tarjeta del dataset, `user_data.labeling`): se calcula con las marcas de la biblioteca y el hardware del spec al confirmar; si el usuario cambia el hardware, se recalcula.
3. **En la página del etiquetado**: una franja "Before you train" (estilo Health Check de Roboflow) o, con la propuesta A, dentro de la tarjeta de cada clase.
4. **Tras entrenar**: el informe ya lleva `small_objects` y `crowded`; pasaría a usar la misma función para no tener dos versiones.

### Cómo se haría (cuando se apruebe)

- `sim2chip/data_checks.py`: función pura `checks(marks, task, classes, hardware) -> [{check, class, text, numbers}]`, sin TensorFlow, que absorbe `counting.small_objects` y `crowded`. Tests con los casos de arriba.
- Servidor: `GET /api/library/labelings/{lid}/checks?hardware=` y un turno de chat de tipo aviso que añade el servidor (no el LLM); el texto también va al contexto del LLM para que no lo contradiga.
- Web: el `Notice` ámbar que ya existe (`ui.tsx`, tono `ember`) como burbuja del chat.
- Coste: un día con tests y prueba de punta a punta.

### Decisiones para el usuario

1. Parte A: ¿B ya, A después y C fuera del hackathon?
2. Umbral de pequeños: 25 % (propuesto) o 50 % (actual).
3. Fundidos en la rejilla: celda vecina (propuesto, como decodifica el modelo) o solo la misma celda (más prudente).
4. Pocas fotos: 10 fuerte y 30 suave; desequilibrio 3 a 1 (sin fuente; ¿medirlo antes?).
5. ¿Incluir los opcionales (cortados por el borde, nota contra el profesor)?
6. ¿Enseñar el "y si" con otro hardware dentro del aviso?
