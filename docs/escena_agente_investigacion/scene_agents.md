# Agentes LLM/VLM para escenas 3D y reconstruccion foto-a-simulacion

Notas para EyeForge (una foto real a escena Isaac Sim para datos sinteticos de detectores pequenos). Fecha de la busqueda: 2026-10-06. Las fichas marcadas "(memoria, no re-verificado hoy)" vienen de conocimiento previo con el enlace arXiv correcto, pero sus cifras no se comprobaron en esta sesion: tratarlas como pendientes de verificar antes de citarlas en publico.

## 1. Ficha por trabajo: entrada, estructura del agente, verificacion, salida, resultados, codigo

### Takeaway
El campo se ha movido de "LLM escribe un layout una vez" (LayoutGPT 2023, Holodeck 2024) a agentes con herramientas y bucle de autocritica multimodal (SceneCraft 2024, SceneWeaver NeurIPS 2025, Scenethesis ICLR 2026) y, en 2025-2026, a reconstruccion desde UNA imagen con agentes que orquestan SAM 3 / SAM 3D / profundidad (SceneConductor 2026, Agentic Real2Sim 2026, Mira-Scene 2026). Nuestro pipeline (Kimi planifica en JSON, SAM 3D, planos con textura de la foto, HDRI) encaja exactamente en esa ultima familia; lo que nos falta frente al estado del arte es el bucle render-critica-arreglo con rollback.

### Cited Findings

**Agentes de texto a escena con bucle de critica**
- SceneCraft (Hu, Iscen, Jain, Kipf, Yue, Ross, Schmid, Fathi; Google; arXiv marzo 2024, ICML 2024): texto a script Python de Blender. Doble bucle: interno, genera grafo de escena con relaciones espaciales, escribe codigo con restricciones numericas, renderiza y un VLM (GPT-4V) critica el render y corrige; externo, "library learning" que compila funciones reutilizables en una biblioteca de habilidades, mejora continua sin reentrenar el LLM. Escala hasta unos 100 assets; supera a otros agentes LLM en adherencia a restricciones y evaluacion humana. [arXiv 2403.01248](https://arxiv.org/abs/2403.01248)
- BlenderAlchemy (Huang, Yang, Guibas; Stanford; abril 2024, rev. agosto 2024; ECCV 2024): edita programas de Blender (materiales procedurales, geometria, iluminacion para renders de producto) con un generador de ediciones y un evaluador de estado basados en VLM (GPT-4V), buscando secuencias de acciones; anade "imagenes de referencia imaginadas" de un modelo texto-imagen para anclar descripciones abstractas. Licencia del paper CC BY 4.0. [arXiv 2404.17672](https://arxiv.org/abs/2404.17672)
- SceneWeaver (NeurIPS 2025): agente reflexivo "reason-act-reflect". Entrada: texto ("Design me a kitchen"). Planificador GPT-4 multimodal que elige herramientas de una interfaz estandarizada por granularidad: inicializadores (data-driven, real-to-sim, LLM), implementadores de microescenas (LLM, guiados por 2D) y refinadores (restricciones, pose, borrado de objetos). Memoria de longitud 1 (ultimo paso, escena y puntuaciones) para evitar alucinaciones. Maximo 10 rondas por escena, parada temprana al estancarse, y rollback/replan si la puntuacion degrada. El evaluador GPT-4 puntua realismo, funcionalidad, correccion del layout y completitud, mas metricas fisicas (colisiones, salidas de limites, numero de objetos). Ejecutor fisico sobre Infinigen y Blender; exporta Blender y USD compatible con Isaac Sim. Resultados en 8 tipos de sala open-vocabulary: 36.5 objetos, 0 colisiones, realismo 8.8, frente a Holodeck 22.3 objetos, 3.9 colisiones, 7.7; I-Design 14.3 objetos, 7.1; LayoutGPT 7.3 objetos, 6.0. Preferencia humana 94.3 % frente a I-Design y 91.4 % frente a Holodeck. Construido sobre OpenManus. Codigo publico. [arXiv HTML 2509.20414](https://arxiv.org/html/2509.20414v1); [GitHub Scene-Weaver/SceneWeaver](https://github.com/Scene-Weaver/SceneWeaver)
- Scenethesis (NVIDIA Cosmos Lab; arXiv mayo 2025; ICLR 2026): marco agentico sin entrenamiento. LLM hace planificacion gruesa (categorias de objetos y prompt "upsampled" con jerarquia espacial); modulo de vision genera una imagen detallada como guia con modelos fundacionales, extrae grafo de escena con poses y recupera assets 3D y un mapa de entorno; optimizacion fisica iterativa (alineacion de pose, sin penetraciones, estabilidad); un juez verifica coherencia espacial y dispara reparacion dirigida, repitiendo hasta cumplir. Escenas interiores y exteriores, menos colisiones y fallos de estabilidad que el SOTA. Codigo no mencionado en la pagina. [Pagina NVIDIA](https://research.nvidia.com/labs/cosmos-lab/scenethesis/); [arXiv 2505.02836](https://arxiv.org/pdf/2505.02836); [ICLR 2026 en mlanthology](https://mlanthology.org/iclr/2026/ling2026iclr-scenethesis/)
- LayoutVLM (Sun, Liu, Gu, Lim, Bhat, Tombari, Li, Haber, Wu; CVPR 2025): un VLM, a partir de imagenes con marcas visuales, genera dos representaciones que se refuerzan (poses iniciales y relaciones espaciales) y se resuelven en una optimizacion diferenciable; "self-consistent decoding" para mejorar la planificacion espacial; afinar el VLM con layouts extraidos de datasets mejora. Codigo publico. [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_LayoutVLM_Differentiable_Optimization_of_3D_Layout_via_Vision-Language_Models_CVPR_2025_paper.html); [GitHub sunfanyunn/LayoutVLM](https://github.com/sunfanyunn/LayoutVLM)
- SceneOrchestra (2026): sintesis agentica de escenas generando la trayectoria completa de llamadas a herramientas de una vez, para eficiencia frente a agentes paso a paso. Solo visto en resultados de busqueda, no leido. [arXiv 2604.19907](https://arxiv.org/pdf/2604.19907)
- VULCAN (diciembre 2025): multiagente con herramientas para colocar objetos 3D iterativamente. Solo titulo verificado. [arXiv 2512.22351](https://arxiv.org/pdf/2512.22351)

**Generadores LLM de escena sin bucle visual fuerte (generacion 2023-2024)**
- LayoutGPT (Feng et al., NeurIPS 2023): LLM genera layouts 2D y 3D en estilo CSS con ejemplos en contexto; sin critica por render. (memoria, no re-verificado hoy) [arXiv 2305.15393](https://arxiv.org/abs/2305.15393)
- Holodeck (Yang et al., CVPR 2024): texto a entorno AI2-THOR; GPT-4 propone suelo/paredes, materiales, puertas/ventanas, objetos de Objaverse y restricciones espaciales que resuelve un solver; entrenaron agentes ObjectNav en escenas Holodeck y mejoraron en tipos de escena nuevos. (memoria, no re-verificado hoy) [arXiv 2312.09067](https://arxiv.org/abs/2312.09067)
- I-Design (2024): varios agentes LLM conversan para disenar una habitacion, grafo de relaciones, backtracking para colocar y recuperacion de assets de Objaverse. (memoria) [arXiv 2404.02838](https://arxiv.org/abs/2404.02838)
- 3D-GPT (Sun et al., 2023): agentes LLM (despachador, modelado de tareas, ejecucion) que llaman funciones procedurales de Infinigen; escenas procedurales (naturaleza), sin assets de foto. (memoria) [arXiv 2310.12945](https://arxiv.org/abs/2310.12945)
- Infinigen (CVPR 2023) e Infinigen Indoors (CVPR 2024): generacion procedural pura (sin LLM), todo desde codigo, licencia BSD; Indoors exporta a simuladores y tiene un solver de restricciones que SceneWeaver reutiliza. (memoria) [arXiv 2306.09310](https://arxiv.org/abs/2306.09310); [arXiv 2406.11824](https://arxiv.org/abs/2406.11824)
- Architect (NeurIPS 2024): escenas interactivas jerarquicas guiadas por inpainting 2D con difusion y profundidad, en vez de pedir coordenadas al LLM. (memoria) [arXiv 2411.09823](https://arxiv.org/abs/2411.09823)

**Real a simulacion desde una o pocas observaciones**
- ACDC, Automated Creation of Digital Cousins (Dai, Wong, Jiang, Wang, Gokmen, Zhang, Wu, Fei-Fei; CoRL 2024): desde UNA imagen RGB construye automaticamente varias escenas "primas" (geometria y semantica parecidas, no copia exacta) recuperando assets de BEHAVIOR; politicas entrenadas en primos: 90 % de exito zero-shot sim-to-real frente a 25 % entrenando en el gemelo digital. [arXiv 2410.07408](https://arxiv.org/abs/2410.07408)
- GRS (2024): de una sola observacion RGB-D crea gemelo digital con tareas resolubles; tres etapas: comprension con SAM2 y VLM, emparejado con assets listos para simulacion, generacion de tareas. [arXiv 2410.15536](https://arxiv.org/html/2410.15536v3)
- SceneConductor (Kim, Lan, Chen, Nguyen, Pan, Pan; junio 2026): escena 3D desde una sola imagen con orquestacion multiagente; usa SAM 3D, profundidad (MoGe), image-to-3D; VLMs citados Claude y Qwen; se compara con SceneWeaver y VIGA en 3D-FUTURE, ScanNet y MIT-67. Numero de rondas y metricas no extraidas (PDF comprimido). [arXiv 2606.08402](https://arxiv.org/pdf/2606.08402); proyecto jhkim0759.github.io/projects/SceneConductor
- Agentic Real2Sim (Chen et al.; UBC, JHU, NUS, Columbia, UCLA; julio 2026): cuatro agentes VLM convierten episodios reales (DROID) en gemelos simulables en MuJoCo. Herramientas SAM 3, SAM 3D, FoundationStereo, FoundationPose. Las consultas al VLM son decisiones acotadas con esquema, no razonamiento abierto. Criticos especificos: "mask critic" que rechaza segmentaciones malas y "tracking critic" de la pose, con presupuesto de reintentos acotado. DROID-100: Gemma 4 31B 48 exitos, 8 parciales, 44 fallos con coste 2.62 $; GPT-5.4 43 exitos con 31.4 veces mas coste; los modelos abiertos rinden comparable. [arXiv 2607.19190](https://arxiv.org/html/2607.19190v1)
- Mira-Scene (VAST-AI-Research; septiembre 2026): reconstruccion composicional editable desde una imagen; "Canonical Coordinate Map" alineado a pixel, transformer de difusion multimodal que co-genera geometria y layout; pipeline: segmentacion interactiva, profundidad, CCM y voxels, malla por objeto, ensamblado y generacion de mapa de entorno. Interiores, exteriores e in-the-wild. Totalmente abierto (inferencia, checkpoints, evaluacion). No es agentico (sin LLM). [arXiv 2609.23796](https://arxiv.org/abs/2609.23796); [GitHub](https://github.com/VAST-AI-Research/Mira-Scene)
- SceneGen (agosto 2025): varios assets 3D y sus posiciones relativas desde una imagen en una sola pasada feedforward. [arXiv 2508.15769](https://arxiv.org/abs/2508.15769)
- Phone2Proc (Deitke et al., CVPR 2023): escaneo con movil (RoomPlan) y generacion procedural de variaciones de la habitacion; ObjectNav sim-to-real de 34.7 % a 70.7 % de exito. (memoria, no re-verificado hoy) [arXiv 2212.04819](https://arxiv.org/abs/2212.04819)
- URDFormer (RSS 2024), Real2Code (ICLR 2025), Video2Game (CVPR 2024): foto(s) o video a escenas articuladas/interactivas (URDF, codigo de articulacion por LLM, juego web con NeRF a malla). (memoria) [URDFormer 2405.11656](https://arxiv.org/abs/2405.11656); [Real2Code 2406.08474](https://arxiv.org/abs/2406.08474); [Video2Game 2404.09833](https://arxiv.org/abs/2404.09833)

**Generacion de tareas de robotica (contexto)**
- GenSim (ICLR 2024) y RoboGen (ICML 2024): LLM propone tareas y codigo de simulacion; RoboGen ciclo propose-generate-learn. (memoria) [GenSim 2310.01361](https://arxiv.org/abs/2310.01361); [RoboGen 2311.01455](https://arxiv.org/abs/2311.01455)

### Inferences
- Lo mas cercano a EyeForge es la combinacion ACDC (una imagen, variaciones "primas" en vez de gemelo exacto) + Agentic Real2Sim (agentes VLM acotados por esquema con criticos por etapa sobre SAM 3 / SAM 3D) + Mira-Scene/SceneConductor (reconstruccion composicional de una foto con mapa de entorno). Ninguno apunta a datos para detectores pequenos ni a exteriores tipo calle; ese hueco es nuestro.
- El diseno de Agentic Real2Sim (decisiones del VLM con esquema JSON, criticos especificos por herramienta, reintentos acotados, modelos abiertos casi tan buenos y 30 veces mas baratos) valida nuestro enfoque Kimi a JSON y apoya usar modelos abiertos.

### Gaps
- No encontre documentacion de "Mira-Scene" como agente; es un modelo de reconstruccion (VAST), no un planificador.
- Licencias de codigo: no verificadas para SceneWeaver, LayoutVLM, Mira-Scene (repos publicos, licencia no leida). Infinigen BSD de memoria.
- Detalles de SceneConductor (rondas, metricas) no extraibles del PDF.

## 2. Bucles iterativos render-critica-arreglo y plantillas procedurales para extender mas alla de la foto

### Takeaway
Los bucles con render y critica VLM existen y son pocos rondas: SceneWeaver hasta 10 con rollback, SceneCraft y BlenderAlchemy iteran con GPT-4V, Scenethesis repite juez-reparacion hasta cumplir, Agentic Real2Sim usa criticos por etapa con reintentos acotados. Para extender mas alla de lo observado, la tradicion es procedural (Infinigen, Phone2Proc, ACDC con primos), no copias rotadas como nuestro cruce.

### Cited Findings
- SceneWeaver: maximo 10 rondas por escena, parada temprana y rollback si la puntuacion multimodal empeora; memoria de un paso. [arXiv 2509.20414](https://arxiv.org/html/2509.20414v1)
- SceneCraft: bucle interno de render y critica con GPT-V para refinar; bucle externo de biblioteca de habilidades. [arXiv 2403.01248](https://arxiv.org/abs/2403.01248)
- BlenderAlchemy: generador de ediciones y evaluador de estado VLM buscando secuencias de ediciones sobre el programa de Blender (material, luz). [arXiv 2404.17672](https://arxiv.org/abs/2404.17672)
- Scenethesis: juez de coherencia espacial que dispara reparacion dirigida y repite. [NVIDIA](https://research.nvidia.com/labs/cosmos-lab/scenethesis/)
- Agentic Real2Sim: "mask critic" y "tracking critic" con presupuesto de reintentos acotado por etapa. [arXiv 2607.19190](https://arxiv.org/html/2607.19190v1)
- ACDC genera varios primos de una sola foto en lugar de un gemelo; Phone2Proc genera variaciones procedurales de un escaneo. [ACDC](https://arxiv.org/abs/2410.07408); [Phone2Proc](https://arxiv.org/abs/2212.04819) (cifras de memoria)

### Inferences
- Patron recomendable para nosotros: criticos por etapa con esquema (suelo/paredes, mallas SAM 3D, HDRI/luz, camara) con 1 a 3 reintentos, y un critico global de render con tope tipo SceneWeaver (unas 5-10 rondas) y rollback si empeora. Es lo que hacen los mejores sistemas, y es barato si el critico es un VLM abierto.
- Nuestro cruce por copias rotadas con ladrillos recoloreados es una forma de "digital cousin" procedural; vale la pena presentarlo con ese marco (ACDC) en el pitch.

### Gaps
- No encontre ningun trabajo que publique ablaciones de "numero de rondas vs calidad final" mas alla del tope de SceneWeaver.
- No encontre trabajos que extiendan una esquina de calle a un cruce por simetria; ese caso concreto parece no publicado.

## 3. Evidencia de que estas escenas sinteticas mejoran el entrenamiento real (detectores, robots)

### Takeaway
La evidencia de ganancia sim-to-real existe para politicas de robot y navegacion (ACDC 90 % vs 25 %, Phone2Proc, Holodeck), y para detectores entrenados con Isaac Sim + Replicator con aleatorizacion masiva (Synthetica), pero no encontre ningun estudio que mida ganancias de un detector entrenado con escenas construidas por un agente LLM/VLM desde una foto. Es un hueco publicable y la pregunta que nuestras pruebas (HOPE) responden.

### Cited Findings
- ACDC: politicas entrenadas en primos digitales 90 % de exito zero-shot sim-to-real frente a 25 % con el gemelo. [arXiv 2410.07408](https://arxiv.org/abs/2410.07408)
- Synthetica (NVIDIA, octubre 2024): Isaac Sim + Replicator con trazado de rayos, 2.7 millones de imagenes, aleatorizacion de materiales, luz y post-proceso; RT-DETR; con sim+real mAP 0.885 en YCB-V y 0.890 en T-LESS, 50-100 Hz. No usa LLMs. [arXiv 2410.21153](https://arxiv.org/html/2410.21153v1)
- La brecha sim2real de detectores se divide en brecha de apariencia y de contenido (elementos y disposicion de la escena), y mezclar sintetico con real da mejores resultados que solo sintetico. [Revision Springer 2025](https://link.springer.com/article/10.1007/s10462-025-11431-3); [PMC industrial](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11154516/)
- Meta-Sim (NVIDIA 2019) aprende a generar distribuciones de escena para reducir la brecha de contenido, antecedente sin LLM. [arXiv 1904.11621](https://arxiv.org/pdf/1904.11621)
- Phone2Proc: ObjectNav real de 34.7 % a 70.7 % (memoria, no re-verificado hoy). [arXiv 2212.04819](https://arxiv.org/abs/2212.04819)

### Inferences
- Los agentes de escena reducen sobre todo la brecha de contenido (que objetos, donde); la de apariencia sigue dependiendo de render y aleatorizacion (Synthetica). Encaja con nuestro hallazgo de HOPE: el sintetico revisado ayuda y el no revisado perjudica, lo que justifica el QC por VLM.

### Gaps
- No encontre benchmarks de detectores (mAP, F1) entrenados con escenas de SceneWeaver, Holodeck, Scenethesis o similares; esos papers miden calidad visual/fisica, no utilidad para entrenar.
- Las cifras de Holodeck en ObjectNav no se verificaron en esta sesion.
