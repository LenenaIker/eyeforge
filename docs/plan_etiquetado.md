# Plan: etiquetados (refactor 2026-10-10)

Esquema aprobado por el usuario el 2026-10-10 (artifact "Esquema de etiquetas EyeForge", versión 3). Los datos de usuario se borraron ese día: no hay migración, la biblioteca nace con el esquema nuevo.

## Idea

- Dataset: carpeta de fotos. Etiquetado: lista de clases que buscar en un dataset, con sus marcas. Modelo: se entrena con un etiquetado.
- Una sola fuente de verdad, la biblioteca (`library.db`). El trabajo recibe una foto fija del etiquetado (`input/labeling.json`, solo lectura) y devuelve las marcas del profesor, que se importan una vez. Nada se copia en sentido contrario.
- El CLI y los bancos de `experiments/` siguen funcionando con carpetas (`user_data.raw`, `labeled`, `class_map`): si el trabajo no trae `labeling.json`, la etapa de etiquetado trabaja como hasta ahora sobre la carpeta.

## Tablas (library.db, VERSION 4)

| Tabla | Columnas | Notas |
|---|---|---|
| dataset | id, owner, name, created | |
| photo | id, dataset_id, content_hash, file, added | el hash solo para la caché del profesor |
| labeling | id, dataset_id, name, task (count, classify), threshold, copied_from, created | |
| class | id, labeling_id, key, description, color, position | key en inglés snake_case, única por etiquetado |
| class_name | id, class_id, phrase, rank | frases para SAM 3 |
| review | labeling_id, photo_id, reviewed_at | PK (labeling_id, photo_id) |
| mark | id, photo_id, class_id, x, y, w, h, score, source | centro y tamaño de caja (w, h nulos en un clic del usuario; todo nulo en clasificación); source teacher o user |
| model | job, labeling_id, trained_at, reviewed_photos | qué etiquetado y cuántas revisadas al entrenar |

Se quedan sin cambios: objects, scenes, kits, uses, removed, profiles y las descripciones de escena por foto (hoy `notes` kind scene; pasa a una tabla `photo_description` propia).

Desaparecen: `notes` de boxes, points y class, `label_edits`, `photos.label`, `photos.split`, `synced` para etiquetas.

## Reglas en código

1. Un objeto, una clase: al guardar marcas de un etiquetado, dos cajas de clases distintas con IoU >= 0.7 se rechazan (las del usuario) o se desempatan con SigLIP 2 (las del profesor, como hoy).
2. Editar clases en el sitio si ningún modelo usa el etiquetado; si alguno lo usa, copia automática "(2)" con las marcas de las clases que siguen. Añadir una clase quita la revisión de las fotos revisadas (el usuario no la buscó en ellas).
3. Estado de cada foto: revisada (fila en review), del profesor (tiene marcas teacher y no está revisada), pendiente (ni lo uno ni lo otro). El profesor nunca escribe en fotos revisadas.
4. El profesor solo corre en fotos pendientes; la caché de SAM 3 por foto y frase sigue evitando repetir GPU.

## Flujos

- **Subir fotos en el chat o en la biblioteca**: crea o amplía un dataset.
- **Subir carpetas ya clasificadas**: crea un dataset y un etiquetado de clasificación con marcas del usuario y todas las fotos revisadas.
- **Chat**: la tarjeta de cada dataset lista sus etiquetados (nombre, tarea, clases, revisadas de total). El spec lleva `user_data.labeling` (id o null) y `user_data.dataset`. El servidor resuelve al crear el trabajo: mismo conjunto de clases, usar; distinto, editar o copiar según la regla 2; null, etiquetado nuevo con las clases del spec.
- **Revisión (calibración)**: cada ronda exporta el etiquetado al trabajo, la etapa marca la muestra, las correcciones van directas a la biblioteca (marcas user + review) y la siguiente ronda vuelve a exportar. Al aceptar, el profesor marca el resto y sus marcas se importan.
- **Entrenamiento**: la etapa de etiquetado escribe `verdicts.json` como hoy (salida de etapa); `ingest` lee el estado de cada foto del snapshot para el test (ver decisión A). Al terminar se registra el modelo con su etiquetado.
- **Corregir después de entrenar**: se edita el etiquetado en su página; "Reentrenar" lanza un trabajo nuevo con el mismo etiquetado.

## Web

- Página del dataset con pestañas Etiquetados, Fotos y Modelos. Cada etiquetado es una tarjeta: nombre, tarea, sus clases dentro, barra de revisadas, Abrir, Entrenar y menú (Renombrar, Copiar, Borrar).
- Página del etiquetado: panel de clases (color, descripción, recuento de marcas, añadir, renombrar, borrar), filtros Todas, Por revisar, Revisadas y Sin marcas, rejilla de fotos con cajas (profesor discontinuo, revisadas continuo) y editor con "Revisada, siguiente".
- La revisión de un trabajo (calibración) usa el mismo editor contra el etiquetado.
- La página del modelo dice con qué etiquetado y cuántas fotos revisadas se entrenó.

## Fases

1. Biblioteca: esquema nuevo y funciones (datasets, fotos, etiquetados, clases, marcas, revisiones, modelos) con tests.
2. Puente con los trabajos: exportar `labeling.json`, etapa de etiquetado e ingest leyéndolo, importar marcas del profesor.
3. Servidor: creación de trabajos, calibración, correcciones y endpoints de etiquetados; borrar `corrections.json`, `kept_labels.json`, `corrections_for`, `follow_labeling`, `labeled_classes`, `label_edits`.
4. Chat: tarjeta de dataset con etiquetados, `user_data.labeling`, comprobaciones del spec.
5. Web: páginas de dataset y etiquetado, editor compartido con la revisión.
6. Limpieza de código muerto, tests, e2e local y en Nebius, deploy.

Estimación: unos 2 días.

## Decisiones del usuario (2026-10-10)

- A. Test: primero fotos revisadas; si hay menos de 20, se completa con fotos del profesor y el informe lo dice.
- B. La carpeta de test aparte (`user_data.test`) sale de la web; queda solo para el CLI y los bancos.

## Estado (2026-10-10)

Implementado y probado: biblioteca, etapas, servidor, chat y web (páginas de dataset y de etiquetado). Prueba local de punta a punta con 24 fotos de manzanas de FSC-147: etiquetado con el profesor, revisión de 6, etiquetado completo (871 marcas) y modelo entrenado sin revisión, registrado en su etiquetado. Diferencias con el plan: las fotos para hacer modelos 3D no entran en la biblioteca (se quedan con su trabajo); `corrections.json` solo guarda las decisiones de modelos 3D y las correcciones de trabajos del CLI.
