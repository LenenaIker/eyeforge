# Tareas del usuario (2026-10-12)

Apuntadas antes de que el usuario se fuera. Se trabajan sin su criterio solo las que no piden decisión; las propuestas se le enseñan antes de aplicarlas.

## Hardware

1. **Flashear la ESP32-CAM desde la web** (por la noche, con el usuario). El cable ha llegado y el móvil reconoce la placa en `espressif.github.io/esptool-js/`. Ver memoria `project_web_flash_test` (firmware/esp32cam y tools/board.mjs).

## Web

2. **Menú Assets del aside desplegable.** Ahora es solo un rótulo con Objects y Scenes debajo; que se pueda plegar y desplegar. (hecho, 6b796c5)
3. **Menú de los tres puntos en Objects sale cortado.** Al pulsar "..." en una tarjeta (p. ej. "metal pitcher") el menú (Rename, Delete) aparece por debajo de la tarjeta y queda tapado. Debe verse entero encima de todo. (hecho, ca70584)
4. **Fotos en el chat con texto, como en Claude o WhatsApp.** Al subir fotos se mandan al momento (solo con el número de imágenes) y el chat responde antes de que el usuario pueda decir nada. Quiere: las fotos quedan adjuntas en la caja de texto, escribe algo si quiere y lo manda todo junto. Sin complicarlo más.
5. **Labels y clases de los datasets, visuales y comprensibles para no expertos.** Abstracto: preparar propuestas buscando qué recomiendan los expertos (herramientas de etiquetado, UX para usuarios no técnicos) y enseñárselas antes de tocar nada.
6. **Modelos 3D sin imagen.** En la pestaña "3D models" de un trabajo (botellas) las tarjetas salen en blanco, solo "kept" y "bottle": el usuario no sabe qué es cada modelo. Cada una debe enseñar su vista previa. (parcial, 6129392: enseña nombre y autor; la vista previa de verdad pide decidir entre un endpoint que sirva el GLB para pintarlo en el navegador o correr `preview_assets` en photo2scene)
7. **Bug en "Test results" de un trabajo de contar (74 fotos).** Las tarjetas salen aplastadas: no se ve la foto, solo los puntos naranjas y los anillos blancos en una franja encima de "counted N / real N". Las fotos deberían verse con sus marcas encima. (hecho, 6129392; el 404 de las fotos es de la demo, cuyas carpetas `label` e `ingest` son enlaces fuera del trabajo)

8. **Contadores de las pestañas en "New kit".** "Datasets 0", "Objects 0" y "Scenes 0" no cuentan, aunque la pestaña Objects tiene muchos objetos. Mirar si es el número de elegidos (entonces que se entienda, p. ej. "0 of 12" o que suba al marcar) o si está roto, y arreglarlo. (hecho, 879cb49)

9. **Avisos amarillos en el chat sobre problemas con los datos.** P. ej. "tus objetos a identificar son demasiado pequeños (menos de 8 px)" a la resolución del alumno de su hardware. Comprobaciones calculadas en código sobre las marcas (tamaño, objetos pegados o solapados, clases con pocas fotos, desequilibrio), mostradas como mensaje amarillo en el chat, no generadas por el LLM.
10. **Nemotron poco competente en la entrevista.** Medir con `interview_bench` alternativas: Nemotron Ultra en lugar de Super, o Nemotron solo para sacar el spec estructurado y otro modelo abierto (Kimi K2.6, Qwen) para conversar. Requisito del hackathon: al menos un modelo abierto de NVIDIA, que puede seguir en un papel concreto. Decide el usuario viendo la comparación. Bases (leídas 2026-10-05, `docs/entrega_devpost.md`): Nemotron no es obligatorio como pieza central, pero el criterio 1 (el desempate) valora cómo se usa Nemotron u otros modelos abiertos de NVIDIA, y el formulario tiene 6 preguntas sobre Nemotron; conviene que tenga un papel claro y que la comparación medida sirva para responderlas.

## Pendiente de antes (decisiones del usuario)

- Dosis de sintético por defecto (propuesta: 10 vistas; hoy 20).
- Reparto imitación / extrapolación (propuesta: 90/10 sin medir hasta que una ruta con extrapolación ayude).
- Revisión a ojo de C.5 (`compare/C5_fase3_clases.html`) y de F.2 en el navegador.
- Prueba de punta a punta por la noche.
