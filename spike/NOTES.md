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
