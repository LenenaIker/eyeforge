# Modelos candidatos para el profesor (etiquetado)

Investigación del 2026-09-28. Fuente: respuesta de GPT al prompt de comparación (sin verificar todavía) y prueba propia con 4 fotos de macetas en Token Factory.

## Problema

El profesor actual (GLM-5.3-Flash, VLM de chat en Token Factory) encuentra la mayoría de los objetos, pero sus cajas son flojas y a veces encaja la planta en lugar de la maceta. Un VLM de chat genera coordenadas como texto: no es un buen generador de propuestas cuando hay 20 objetos pequeños por foto. Parte del fallo era nuestro: `COUNT_PROMPT` pedía "a pot and the plant growing in it".

## Prueba en Token Factory (2026-09-28)

Solo 4 modelos aceptan imágenes: GLM-5.3-Flash (el mejor, ~7 s/foto), Kimi-K3 (algo más fino, ~30 s y falla la mitad), Gemma 3 27B (cajas en sitios vacíos) y MiniCPM-V 4.5 (respuestas rotas). Qwen3.5, Qwen3.8 y GLM-5.3 no aceptan imágenes allí.

## Candidatos (según GPT, por verificar)

| Modelo | Tipo | Licencia | Por qué interesa |
|---|---|---|---|
| **SAM 3** (Meta, nov 2025) | Segmentación por concepto (texto o ejemplar) | SAM License (verificada 2026-09-28): uso comercial y modelos derivados permitidos; prohíbe uso militar, nuclear y espionaje e ingeniería inversa; al redistribuir, incluir la licencia y citar SAM al publicar | Diseñado para devolver *todas* las instancias de un concepto; admite ejemplar positivo (una caja de maceta) |
| **LocateAnything-3B** (NVIDIA, may 2026) | VLM de detección densa | NVIDIA License (verificada 2026-09-28): solo "research or evaluation purposes"; uso comercial solo NVIDIA. Ni el hackathon (premios) ni el producto encajan con claridad: usarlo solo para evaluar | Entrenado para detección densa y etiquetado de datasets; es de NVIDIA (encaja con el hackathon) |
| ~~Qwen3-VL-8B~~ (descartado 2026-09-28: falla muchísimo como detector y como verificador) | VLM con localización | Apache 2.0 (verificada 2026-09-28) | RefCOCO ~89.7; buen verificador ("¿es la maceta y no la planta?") |
| **Rex-Omni-3B** (CVPR 2026) | Híbrido detector/VLM | revisar | Comparable a Grounding DINO en COCO/LVIS |
| **Grounding DINO-B** + teselas | Detector texto a cajas | Apache 2.0 | Línea base abierta y reproducible |
| Gemma 4 E4B (mar 2026) | VLM general | Apache 2.0 | Dice soportar detección y señalado, sin números de localización |

## Diseño recomendado

Detector que maximiza la recuperación (SAM 3 o LocateAnything) y VLM que verifica cada candidato con la descripción del usuario y las reglas de la calibración. Teselado (2x2) para objetos de 15 a 30 px. Correr en GPU propia de Nebius AI Cloud (ninguno está en Token Factory): coste estimado por GPT de 0,10 a 0,35 $ por 100 fotos en H100, dominado por el arranque.

## Pendiente

- Licencias de SAM 3, LocateAnything-3B y Qwen3-VL-8B verificadas (2026-09-28). Consecuencia: SAM 3 pasa a ser el detector por defecto; LocateAnything solo como comparación en el benchmark.
- Verificar precios de Nebius (L40S y H100 por hora) y licencia de Rex-Omni.
- Benchmark propio: fotos de macetas con las correcciones de la calibración como verdad; medir recuperación por instancia y confusión planta/maceta.

## Benchmark local (2026-09-28, RTX 3090)

Verdad: las 31 fotos corregidas por el usuario en `maceta_counter_8` (219 macetas, un punto por maceta). Una maceta cuenta como encontrada si una caja contiene su punto (una caja, una maceta). Código en `spike/teacher_bench/`, resultados en `runs/teacher_bench/` (`review/index.html` y una imagen comparativa por foto en `compare/`). Texto de búsqueda sacado del spec (`assets.queries`), sin ajustes por clase.

| Profesor | F1 | Precisión | Recall | s/foto |
|---|---|---|---|---|
| GLM-5.3-Flash (prompt sin la frase de la planta, sin reglas) | 0,74 | 0,83 | 0,68 | 7,6 (en paralelo) |
| SAM 3, umbral 0,5 (elegido sobre estas mismas fotos: optimista) | 0,69 | 0,68 | 0,70 | 1,5 |
| SAM 3, umbral 0,3 | 0,52 | 0,38 | 0,81 | 1,5 |
| LocateAnything-3B | 0,58 | 0,64 | 0,53 | 1,3 |
| LocateAnything + Qwen verificador | 0,48 | 0,73 | 0,35 | |
| LocateAnything teselas 2x2 | 0,55 | 0,42 | 0,77 | 2,9 |
| LocateAnything teselas + Qwen | 0,56 | 0,61 | 0,52 | |
| GLM + Qwen verificador | 0,45 | 0,92 | 0,30 | |
| Qwen3-VL-8B como detector | 0,30 | 0,21 | 0,51 | 21 |

Hallazgos:
- LocateAnything con decodificación voraz entra en bucle (repite la caja de la imagen entera); hay que usar el muestreo de su model card (0,7, top_p 0,9). Sin teselas deja fotos enteras sin marcar; con teselas encuentra casi todas las macetas pequeñas, pero marca ventanas.
- Qwen como verificador sí/no dice que sí a casi todo (27 de 31 cajas falsas). Pedirle que nombre lo que hay en la caja (recorte de la foto original) y juzgar el nombre separa mejor, pero tira demasiadas macetas buenas: baja el F1 en todos los casos.
- Parte de las "falsas" de GLM las nombra Qwen como "flower pot": pueden ser macetas sin marcar o cajas duplicadas. Pendiente de la revisión del usuario.
- SAM 3 (acceso aprobado 2026-09-28): el que más macetas encuentra con umbral bajo (81 %), pero marca ventanas. Con umbral 0,5 queda cerca de GLM, cinco veces más rápido y en local.
- En la foto de la calle, SAM 3 y LocateAnything coinciden en cajas de las jardineras del muro justo por debajo de los puntos del usuario: puede que los puntos estén sobre la planta y no sobre el contenedor. Pendiente de la revisión del usuario.
- Siguiente propuesto: repetir en benchmarks públicos (FSC-147 para conteo con puntos, subconjunto de RF100-VL para detección en varios dominios), con el dataset de macetas como caso de usuario.

### Ronda 2 (2026-09-28): prompts según la guía de cada modelo, ejemplares y umbral por preferencia

Separación: 6 fotos de calibración (repartidas por la carpeta, como la app) y 25 de test (176 macetas). Números solo del test.

| Profesor | Umbral (elegido en calibración) | F1 | F2 | F0,5 | Precisión | Recall |
|---|---|---|---|---|---|---|
| GLM, prompt largo con la descripción | | **0,72** | 0,67 | 0,77 | 0,81 | 0,64 |
| GLM, prompt corto estilo SAM 3 (píxeles) | | 0,71 | 0,63 | **0,80** | 0,88 | 0,59 |
| SAM 3 texto, preferencia "encontrar más" | 0,55 | 0,67 | 0,65 | 0,69 | 0,71 | 0,64 |
| SAM 3 + ejemplares, "encontrar más" | 0,25 | 0,67 | **0,68** | 0,67 | 0,66 | 0,69 |
| SAM 3 + ejemplares, "menos falsos" | 0,5 | 0,63 | 0,55 | 0,75 | 0,85 | 0,50 |
| LocateAnything según model card (`</c>`, 8192 tokens, 1536 px) | | 0,43 | 0,56 | 0,34 | 0,31 | 0,71 |
| Qwen3-VL, prompt del cookbook | | 0,23 | 0,33 | 0,18 | 0,15 | 0,47 |

- GLM trabaja en píxeles: con coordenadas 0..1000 cae a F1 0,17. El prompt corto sube su precisión (0,88) a costa de recall.
- Seguir la guía de LocateAnything sube el recall (0,70) pero hunde la precisión; a 2048 px no cabe en 24 GB.
- Ejemplares de SAM 3 (franja con recortes de las fotos de calibración, positivos en macetas y negativos en ventanas, ejemplo en `runs/teacher_bench/sam3_ex_mosaic_example.jpg`): mejora poco, solo en el modo "encontrar más".
- El umbral elegido en 6 fotos es ruidoso (para SAM 3 texto, "encontrar más" eligió un umbral más alto que "equilibrado"). Con más fotos de calibración o una curva más suave sería estable.
- Al usuario le gusta más visualmente SAM 3 con umbral 0,5, pero por métrica gana GLM: sospecha de que la verdad (puntos) está a veces sobre la planta y no sobre el contenedor. Revisar antes de decidir.

### Decisión del usuario (2026-09-28)

Tras revisar las imágenes: SAM 3 con umbral 0,5 le parece el mejor; GLM con prompt corto, LocateAnything según su guía y SAM 3 con ejemplares (0,4) también dan buenos resultados en general. Qwen3-VL queda fuera de la lista (ni detector ni verificador).

### Profesor elegido: SAM 3 (usuario, 2026-09-28)

SAM 3 pasa a ser el profesor. Motivos: es el que mejor le parece visualmente (umbral 0,5), da una confianza por objeto (el umbral se ajusta a la preferencia de errores del usuario) y devuelve máscaras, que son las que necesitan tres de las cuatro tareas que se ofrecen ahora (conteo, rejilla por zonas y la presencia en clasificación; ver `docs/tipos_de_modelo.md`). La regresión no depende del profesor (su valor sale de Isaac).

## Clasificación de superficies: césped (2026-09-30)

Verdad: las 69 fotos de `real/grass` etiquetadas por el usuario en una página de Claude (`real/grass_labels.json`: 34 grass, 24 border, 11 nograss). Criterio del usuario: robot cortacésped; grass se corta, border exige cuidado con las cuchillas, nograss nunca se usan las cuchillas. Scripts en el scratchpad de la sesión (no en el repo).

| Profesor | Sin ejemplos | 6 fotos del usuario | 12 fotos del usuario |
|---|---|---|---|
| SAM 3 (frases "grass", "edge", "no grass", objeto más seguro) | 0,58 | | |
| GLM-5.3-Flash con la explicación completa del usuario (~2 min/foto) | 0,41 | | |
| DINOv2-base / large + centroides o regresión logística | | 0,65 / 0,60 | 0,70 / 0,68 |
| SigLIP 2 base / large / so400m + centroides | | 0,77 / 0,79 / 0,78 | 0,81 / 0,83 / 0,82 |
| **SigLIP 2 large + centroides + descripción en inglés** | 0,48 | **0,87** (peor 0,79) | **0,87** (peor 0,82) |
| Lo mismo, dentro del pipeline (`label`, 10 sorteos) | 0,48 | 0,83 (peor 0,73) | 0,84 (peor 0,75) |

Medias sobre 20 sorteos de qué fotos corrige el usuario (2 o 4 por clase); el acierto se mide en las demás. Decir siempre "grass" acierta ~0,5.

- SAM 3 busca objetos: casi todos los bordes acaban en grass y las fotos sin hierba en la clase "empty" (sobraba).
- GLM describe bien las fotos pero con la definición escrita de "border" (mezcla de hierba y superficie dura) casi todo es border: el criterio del usuario no se deja escribir bien en una frase.
- Aprender de ejemplos gana: SigLIP 2 con 6 correcciones pasa de 0,48 a 0,83. La descripción en inglés estabiliza los sorteos malos; en español no ayuda (SigLIP entiende mal el español).
- Importa qué fotos se corrigen: la calibración elige ahora fotos lo más distintas posible (`teacher.spread`), no espaciadas por nombre.

### Decisión (usuario, 2026-09-30)

Profesor por tarea, a revisar con pruebas por tipo de objetivo: **SigLIP 2 para clasificar**, **SAM 3 para contar**. Los modelos siempre se la juegan: ni "unsure" ni una clase de fondo inventada cuando las clases cubren toda la foto (`background_class: null`). El Sipeed MaixCAM usa la misma idea (huellas de una MobileNet + centroides) en el propio chip; una exportación "de huellas" del alumno queda como idea.
