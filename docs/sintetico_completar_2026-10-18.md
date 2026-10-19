# El sintético completa, no amplía (2026-10-18)

Punto de partida para la sesión del 2026-10-19. Sale de las medidas del 17 y el 18 y de una búsqueda sobre cómo consigue la industria que el sintético aporte (fuentes al final).

## Lo medido hasta hoy

| Caso | Solo fotos | Fotos + sintético (mezcla) | Primero sintético, luego real |
|---|---|---|---|
| Contenedores, esferas (16 fotos de entrenamiento, test revisado a mano) | **0,611** | 0,541 | 0,550 |
| HOPE, escenas desde la foto revisadas (40 fotos) | 0,819 | **0,830** | 0,816 |
| HOPE, sintético del producto (2000 vistas) | 0,819 | 0,776 | sin medir (las vistas ya no están) |
| Calle, ruta actual (V.6) | | sin ganancia | |

Detalle en `experiments/NOTES.md` (S.6c y "Primero sintético y después real").

## Qué hacen los profesionales

1. **Usan el sintético para cubrir huecos, no para repetir lo que ya hay.**
   - Casos con éxito: clases raras, defectos poco frecuentes, niebla, noche y poses que no salen en las fotos.
   - Cosmos-Drive-Dreams (NVIDIA) sube de 0,299 a 0,337 en detección 3D, y la mayor parte de la ganancia viene de la niebla (+9,4 %). En coches con buen tiempo apenas cambia o baja.
2. **Miden donde está el hueco.** El test incluye las condiciones que el sintético viene a cubrir. Con un test igual que el entrenamiento, el sintético no tiene nada que añadir.
3. **Trabajan en bucle guiado por errores.** Entrenan, miran dónde falla el modelo en datos reales y cambian la generación para cubrir ese fallo. Repiten. NVIDIA, en su ejemplo de carretillas de almacén, cierra la diferencia en tres vueltas.
4. **Buscan variedad estructurada, no fotorrealismo.** Colocan los objetos donde tienen sentido y varían mucho fondo, luz, textura y distractores (Structured Domain Randomization, Prakash et al.). Realismo y variedad a veces chocan.
5. **Cubren las poses de forma sistemática.** Hinterstoisser et al. enseñan cada objeto en todas las poses y condiciones, de fácil a difícil, y así ganan a muestrear al azar. El fondo pesa mucho.
6. **Cortan y pegan recortes reales** (Dwibedi et al., "Cut, Paste and Learn").
   - Pegan objetos reales recortados sobre otros fondos, con varios modos de fusión.
   - Combinado con las fotos reales, ganan más de un 21 % relativo; con el 10 % de las reales superan a usarlas todas.
   - No hace falta modelo 3D.
7. **Preentrenan con sintético y ajustan con real.** Es la receta habitual cuando hay mucho sintético. Con nuestras dosis (45 y 133 vistas) no ganó.

## Lo que significa para nosotros

### Qué le vendemos al cliente

- **Lo que no prometemos:** "más datos de lo tuyo". Sus fotos ya cubren su sitio, su luz y su cámara mejor que cualquier render, y las medidas lo confirman.
- **Lo que sí prometemos:** **completar** su dataset con lo que sus fotos no tienen:
  - otros sitios donde irá la cámara;
  - otra hora o tiempo (noche, lluvia, sol bajo);
  - poses y distancias que no fotografió;
  - casos raros que casi nunca verá pero que importan (un animal concreto, un objeto caído, un cubo en otro soporte).
- **El mensaje:** "Tus fotos enseñan lo normal; el sintético cubre lo que aún no has visto." Encaja con la generalización, que es lo que más valora el jurado según `CLAUDE.md`, y con decir "pocas fotos reales + sintético".
- **En la entrevista:** preguntar qué condiciones faltan en sus fotos (lugares, horas, tiempo, casos raros). Ese hueco es lo que genera el sintético, y el spec lo guarda.

### Por qué nuestras medidas no lo reflejan

- El test de contenedores son fotos de las mismas calles, el mismo día y con la misma luz que las de entrenamiento. Es justo el caso en que el sintético no puede aportar. Lo mismo pasa con HOPE (mismos entornos en entrenamiento y test).
- Estamos midiendo "más de lo mismo", que es justo lo que no vendemos.

### El bucle de errores que ya tenemos y lo que le falta

- **Lo que hay:**
  - El usuario corrige las marcas del profesor, y el profesor aprende su criterio (SigLIP 2 con pocas correcciones, umbral de SAM 3 según los errores que prefiere).
  - Se reentrena con las correcciones.
  - El VLM revisa las vistas sintéticas y el filtro de parecido las compara con las fotos.
- **Lo que falta:** que **los fallos del alumno guíen qué se genera**. Hoy la generación no mira dónde falla el modelo. La validación real ya dice, por clase y por foto, qué se pierde (recall bajo en una clase, objetos pequeños, lejanos o colgados). Esa información debería volver a la generación como cuota de vistas: más de lo que falla y menos de lo que ya acierta.

## Qué mejorar, por orden

Plazo: entrega el 30 y el vídeo es prioridad. Los puntos 1 y 2 son baratos y cambian el discurso; el resto, solo si cabe.

1. **Test que mida el hueco** (unas 2 h, sin renders).
   - Partir los contenedores por sitio (GPS o agrupación de las fotos) para que el test sea de calles que el entrenamiento no vio.
   - Medir otra vez solo fotos frente a fotos + esferas.
   - Si el sintético gana ahí, es la prueba que necesitamos para el vídeo; si no, lo sabemos con datos.
   - Hay que hacer lo mismo con un caso de luz distinta, si tenemos fotos de otra hora.
   - **Preparado el 2026-10-19, sin ejecutar.**
     - Las 40 fotos tienen todas el mismo GPS (un fijo que no se actualizó), así que el sitio sale del orden del paseo (14:47 a 15:23).
     - `bins_job.py --name bins_place --test-from 150100`: 25 fotos de entrenamiento (14:47-15:00) y 15 de test (15:01-15:23). Escribe `ingest/split.json`, que ingest respeta. El trabajo está creado en `p2s_test/jobs/bins_place`.
     - Hoja por tramos para confirmar que no se repiten calles: `data/sphere_test/bins_places_by_time.jpg`. Los pilares de ladrillo con cubos colgados de 14:52 y de 15:10 podrían ser la misma manzana.
     - **Bloqueo:** `p2s_test/jobs/` ya no existe (la carpeta se modificó el 2026-10-18 a las 17:43, probablemente al liberar espacio para Matrix-3D). Con ella se perdieron `bins_count`, `roads_cars` y la verdad de test revisada a mano (`points_human.json`). Solo queda `bins_train/context_agreed.json` (13 fotos de entrenamiento con cajas). Hace falta etiquetar con SAM 3 y que el usuario revise las 15 de test antes de medir.
     - Brazos: `s8_arms.py --job bins_place --keep-ingest`.
2. **El hueco en el spec y en la entrevista** (unas 2 h).
   - Una pregunta de la entrevista: qué falta en tus fotos.
   - Un campo en el spec con los lugares, condiciones y casos que hay que cubrir.
   - La generación los usa como cuota (esferas de otros sitios, luz de noche o de lluvia).
   - Así el discurso "completar" queda dentro del producto y no solo en el vídeo.
3. **Cortar y pegar recortes reales del usuario** (unas 4 h).
   - SAM 3 ya da las máscaras de sus objetos. Se pegan sobre:
     - las esferas, con la colocación por contexto que ya existe;
     - y sobre sus propias fotos, en zonas de suelo libres.
   - Se usan varios modos de fusión, y la escala sale de la profundidad de MoGe.
   - Evita el problema de los modelos 3D (cubos pequeños, coches raros), que es nuestro fallo principal de aspecto y de forma.
4. **Bucle guiado por los fallos del alumno** (medio día).
   - Tras el primer entrenamiento, la validación real da el recall por clase y por tipo de objeto (tamaño, distancia, colgado o de pie, con el contexto de `sphere/context.py`).
   - La siguiente ronda de generación reparte las vistas según esos fallos.
   - Es el punto 3 de la industria y encaja con la GPU caliente (rondas).
5. **Calidad de los modelos 3D** (continuo). Revisión más dura para los objetivos de cerca (QA realista, filtro de parecido con los recortes del usuario). Si una clase no tiene buenos modelos, se recurre al punto 3 y no a un modelo malo.
6. **Dosis.** Mantener la proporción sintético/real en la zona que funcionó (entre 1 y 4 sintéticas por real, HOPE BS). El código de colores de la web ya lo enseña. Si alguna vez generamos mucho más, probar `training.schedule: "synth_then_real"` (ya existe, apagado por defecto).

## Fuera por ahora

- Esferas con sol: no habrá días de sol antes de la entrega.
- Cosmos Transfer para cambiar el tiempo o la hora de las vistas: es justo "completar" (NVIDIA lo mide con niebla y lluvia), pero sale caro de GPU y está sin probar con nosotros. Para el vídeo puede mencionarse como siguiente paso.
- SAM 3D para los objetos borrados: la medida del 16 dice que empeora.

## Fuentes

- [NVIDIA, carretillas de almacén con datos sintéticos (bucle guiado por errores)](https://developer.nvidia.com/blog/how-to-train-autonomous-mobile-robots-to-detect-warehouse-pallet-jacks-using-synthetic-data/)
- [Cosmos-Drive-Dreams (NVIDIA): la ganancia está en la niebla](https://research.nvidia.com/labs/toronto-ai/cosmos_drive_dreams)
- [Cosmos Transfer, aumento por tiempo atmosférico (Cookbook)](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer1/inference-its-weather-augmentation/inference.html)
- [Structured Domain Randomization (Prakash et al., ICRA 2019)](https://arxiv.org/pdf/1810.10093)
- [An Annotation Saved is an Annotation Earned (Hinterstoisser et al., ICCVW 2019)](https://arxiv.org/pdf/1902.09967)
- [Cut, Paste and Learn (Dwibedi et al., ICCV 2017)](https://www.ri.cmu.edu/?p=103478)
- [Simple Copy-Paste (Ghiasi et al., CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf)
- [Meta-Sim: hueco de aspecto frente a hueco de contenido (Kar et al., ICCV 2019)](https://arxiv.org/pdf/1904.11621)
- [Realismo frente a aleatorización (MDPI Sensors 2021)](https://www.mdpi.com/1424-8220/21/23/7901)
- [Aleatorización en fabricación (2025)](https://www.researchgate.net/publication/392530224_Domain_Randomization_for_Object_Detection_in_Manufacturing_Applications_using_Synthetic_Data_A_Comprehensive_Study)
- [Domain randomization: preentrenar con sintético y ajustar con real](https://www.emergentmind.com/topics/domain-randomization-techniques)
