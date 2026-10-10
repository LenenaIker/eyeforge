# Plan: photo2scene en el producto (2026-10-11)

El usuario decidió el 2026-10-11 integrar photo2scene. Los assets de internet dejan de ser objetos del usuario y quedan en nuestro catálogo interno (`asset_db.py` y la caché). Esa parte ya está hecha (`5c321c0`, biblioteca versión 5).

## Idea

- Una escena es un lugar reconstruido a partir de una foto del usuario: el fondo con los objetos borrados, la profundidad, el soporte o la calle, la luz medida y los objetos borrados convertidos en modelos 3D.
- Construirla es caro (de 6 a 23 min por foto en la 3090, según `flow.json` de las pruebas). Reutilizarla es barato: cada trabajo pide composiciones nuevas sobre la misma escena (en la ruta de soporte, unos 3 min con S.5 y S.6).
- Por eso la escena se guarda en la biblioteca del usuario, ligada a la foto de la que sale, y los trabajos siguientes con ese dataset la reutilizan.
- Lo que se borra de la foto pasa a ser objeto del usuario (`source = "scene"`). Los de las clases objetivo sirven como modelos de esas clases: son los objetos reales del usuario, no parecidos de internet.
- Si la escena falla, el trabajo sigue con la plantilla, como ya pasa con todo lo no esencial (`Stage.skippable`), y el informe dice por qué.

## Tablas (library.db, versión 6)

| Tabla | Cambio | Notas |
|---|---|---|
| scenes | + photo_id (FK photo, SET NULL), route (support, street, room), status (ready, failed), why | `files` pasa a ser la carpeta de la escena; una por foto y dueño |
| objects | + scene_id (FK scenes, SET NULL) | los objetos borrados de una escena; borrar la escena no borra sus objetos |

Una escena fallida queda apuntada (status failed y por qué) para no reintentar la misma foto en cada trabajo. La web no la enseña como escena.

## Flujo en un trabajo

1. Nueva etapa `scene_from_photo` (runtime isaac, la misma GPU que el render), después de `label` y antes de `render`. Solo si el spec tiene sintético y fotos del usuario.
2. Escenas del dataset ya listas: se usan tal cual. Si no hay ninguna, construye una con la foto más típica del dataset (la que más se parece al perfil). Tope de tiempo y fallo saltable.
3. Recibe las clases del spec (`--targets`) y la ruta que el análisis de fotos sugiere (mesa o suelo de cerca, calle, habitación).
4. Al acabar, la biblioteca toma la escena y sus objetos (`take_scenes`, como `take_teacher`).
5. `render` mezcla fotogramas de la escena (composiciones nuevas con las clases del spec) y de la plantilla. Con escena, la mayoría salen de ella; el reparto exacto se mide en la prueba.

## Código

- Según la auditoría de diseño (C.11), el producto no importa nada de `experiments/`. Lo que usa la ruta se copia a `sim2chip/photo2scene/` (orquestador con un entorno por paso, como el experimento). Lo que no usa se queda donde está.
- Entornos en Nebius: kits nuevos con el mismo sistema que hoy (tar zstd en el bucket de caché). El de mallas (SAM 3D Objects y MoGe) y el de borrado (Qwen-Image-Edit 2511 con Lightning y el LoRA Object-Remover). Los pesos van en el bucket como los de los profesores. Los pasos de geometría (trimesh, pyrender) van en el kit de mallas para no tener un quinto entorno.
- Las preguntas al VLM (observar, plan, revisión) van a Token Factory como hoy.

## Fases

1. Biblioteca: tablas, `take_scenes`, objetos con escena, tests.
2. Ruta de soporte en el producto, en local (3090): paquete `sim2chip/photo2scene/`, etapa `scene_from_photo`, render con fotogramas de la escena.
3. Prueba local con los datasets de "water container counter" y una mesa de las pruebas: F1 con y sin escena.
4. Ruta de calle (fases 2 y 3 del experimento) y de habitación.
5. Nebius: kits, pesos, tiempos y coste en L40S, prueba de punta a punta.
6. Web: página Scenes con sus objetos, la escena en la página del trabajo y en el chat. Deploy junto con `600d872`.

## Decisiones abiertas (propuesta por defecto)

- A. Cuántas escenas por trabajo: una si el dataset no tiene ninguna; después se reutiliza. Más adelante, un botón "Rebuild this photo as a scene" en el dataset.
- B. Cuándo se construye: dentro del trabajo, no al subir las fotos. Al subir costaría GPU aunque el usuario no llegue a entrenar.
- C. Borrado en Nebius: Qwen en bf16 si cabe en los 48 GB de la L40S; si no, nf4 como en la 3090. Se mide en la fase 5.
