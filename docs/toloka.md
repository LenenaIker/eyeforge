# Toloka y Tendem: encaje con EyeForge

Nota del 2026-10-05 (pista del usuario: Nebius, además de Tavily, tiene Toloka). Solo investigación, nada integrado. Decisión del usuario (2026-10-05): el etiquetado por personas queda aparcado hasta que EyeForge ofrezca servicios para hardware más avanzado.

## Qué es

- **Toloka**: empresa de datos para IA con sede en Ámsterdam, filial de Nebius Group; en mayo de 2025 Bezos Expeditions lideró una ronda de 72 M$. Anotación y evaluación con personas (human-in-the-loop). https://en.wikipedia.org/wiki/Toloka
- **Plataforma de Toloka** (https://toloka.ai/platform): autoservicio en el que un agente monta el flujo de recogida y anotación a partir de la descripción del objetivo. Sin mínimos ni contratos, "you only pay for work that passes QA"; precio fijado tras definir el proyecto, no publicado. Plazos: "Simple classification tasks with the General expert tier can be completed in 24-48 hours". Hay trabajo de imagen y vídeo, pero la página destaca sobre todo tareas de texto y evaluación de LLM.
- **Tendem** (https://nebius.com/solutions/tendem, anuncio del 2026-02-26 https://nebius.com/blog/posts/nebius-and-toloka-to-introduce-integration-to-bring-human-experts-on-demand-to-ai-agents): expertos humanos como "llamada de API de alta latencia y alta precisión" para agentes, por MCP. Más de 10.000 expertos verificados en más de 20 dominios. Nebius lo presenta como tercera capa junto a Token Factory (inteligencia) y Tavily (búsqueda). Tendem funciona como plataforma propia; la integración MCP con Nebius está en acceso anticipado. No publica precios ni plazos, ni menciona tareas de visión.
- La página del hackathon no menciona Toloka ni Tendem (sin créditos ni premio propio; los de socios son Tavily y Jetson de NVIDIA).

## Dónde podría encajar

1. **Test honesto.** Hoy varias cifras de test usan marcas de SAM 3 sin revisar (mesa, macetas). Un conjunto de test verificado por personas daría cifras defendibles ante el jurado. Es un uso puntual, sin prisa: los 24-48 h no molestan.
2. **Distinciones finas que el profesor no sabe hacer** (especies casi idénticas, defectos de piezas): Tendem con un experto del dominio para etiquetar unas pocas fotos difíciles, como opción premium para el usuario. Encaja con el pitch ("los fundacionales fallan en distinciones finas") y con la regla de usar modelos abiertos primero: solo donde el profesor falle, medido.
3. **Preguntas ambiguas de la entrevista** (Tendem por MCP): posible, pero añade horas a algo que hoy es interactivo; poco valor para la demo.

## Lo que no encaja

- El flujo principal promete un modelo en minutos; anotación humana de 24-48 h no cabe ahí, solo como opción aparte o para pruebas internas.
- Sin precios públicos ni créditos del hackathon: probarlo cuesta dinero propio.

## Siguiente paso posible (a decidir)

Pedir a Toloka presupuesto y plazo para revisar el test de un caso (por ejemplo las 50 fotos de la mesa, puntos por objeto) y comparar con las marcas de SAM 3. Si la diferencia es grande, también sería un buen hallazgo para `docs/feedback_sponsors.md`.
