# Feedback para Nebius, NVIDIA y Tavily

Notas para el premio al mejor feedback del hackathon (100 €). Aquí se apunta todo lo curioso que encontramos usando sus productos: fallos, sorpresas, datos medidos y propuestas. Cada punto con fecha y, si se puede, la cifra o el modo de reproducirlo. Al final se redacta en inglés para enviarlo.

Criterio: hechos medidos o vistos por nosotros, no opiniones sueltas. Si algo es una suposición, se dice.

## Nebius AI Cloud (Serverless AI Jobs)

### Rendimiento y arranque

- **El registro y Object Storage dan la misma velocidad dentro de un trabajo, unos 60 MB/s por descarga** (2026-10-04). Esperábamos que el bucket, en la misma región, fuera mucho más rápido (una medida suelta con los pesos de SAM 3 dio 3,4 GB en 7 s), y sacamos los entornos Python de las imágenes para bajarlos del bucket. Medido con detalle: 58-63 MB/s por fichero aun con 16 conexiones (aria2c), y unos 120 MB/s sumando dos descargas en paralelo. Además, descomprimir 2,2 GB de zstd tarda 34 s en el nodo (1gpu-8vcpu-32gb) frente a 6 s en un PC de 24 hilos: parece que el disco del contenedor o la CPU limitan. Propuesta: publicar el ancho de banda esperado de red y disco por preset, y una vía rápida (caché local o disco más rápido) para imágenes y datos grandes.
- **8 vCPU por L40S se quedan cortos para entrenar detectores** (2026-10-04). El mismo entrenamiento de PicoDet-S (900 pasos, 10 fotos, 3 semillas) tarda 11,4 min en `1gpu-8vcpu-32gb` frente a 7,9 min en una RTX 3090 de escritorio con 24 hilos; la carga y el aumento de datos de PaddleDetection corren en CPU y la L40S espera. Encaja con lo de zstd de arriba. Propuesta: un preset de L40S con más vCPU para entrenamiento, o decir en la documentación qué preset conviene para cargas con mucho preprocesado en CPU.
- **Los nodos no guardan imágenes en caché** (2026-10-04). Cada trabajo vuelve a bajar la imagen entera, aunque el trabajo anterior usara la misma en el mismo tipo de nodo. El tiempo es proporcional al tamaño comprimido. Con la imagen de Isaac Sim (13,2 GB) son 3,5-4 min de GPU pagada sin hacer nada. Propuesta: caché de imágenes en los nodos, o al menos de las imágenes públicas de NVIDIA más usadas (Isaac Sim, PyTorch).
- **La descarga de imagen se cobra como tiempo de GPU** (deducido de los tiempos de la ejecución y su coste; sin confirmar en la factura). En nuestras ejecuciones cortas, el arranque es un tercio del total. Propuesta: no cobrar la GPU hasta que el contenedor arranca, o mostrar en la consola el desglose de fases (provisioning, starting, image_pulling, running), que la API ya da.
- **Fases del trabajo bien expuestas** (positivo). `provisioning`, `starting`, `image_pulling` y `running` se pueden seguir desde la CLI, y con eso medimos todo lo anterior.

### Almacenamiento

- **El montaje de buckets (`--volume`, mountpoint-s3) solo funciona como root y no permite renombrar ni reescribir** (2026-09). La imagen oficial de Isaac Sim corre como usuario `isaac-sim` (uid 1234, sin sudo), así que no puede usarlo, y los descargadores que escriben un `.part` y lo renombran dejan ficheros de 0 bytes. Tuvimos que bajar y subir con la API de S3. Propuesta: opción de uid/gid en el montaje y documentar las limitaciones de mountpoint-s3.

### CLI y API

- **`nebius ai job logs` solo devuelve las últimas ~100 líneas** (2026-09). Para un trabajo de 15 min no basta: el worker sube su propio log al bucket cada 20 s. Propuesta: paginación o `--follow` completo.
- **`--inject-file` tiene un límite de 64 KiB y es de solo lectura** (2026-09). Suficiente para un script, pero no está documentado de forma visible.
- **`--format json` en los comandos `create` no imprime nada parseable** (2026-09). Hay que hacer `get-by-name` después para saber el ID.
- **Login interactivo en el navegador** (2026-09-29). Al lanzar trabajos desde un servidor se abría una ventana del navegador para iniciar sesión (probablemente la CLI con perfil de usuario renovando el token; no lo verificamos). Lo resolvimos con una cuenta de servicio con clave autorizada (2026-10-04), pero costó encontrar cómo. Propuesta: que la documentación de Serverless Jobs ponga la cuenta de servicio como el camino para servidores y CI.

### Cuenta y cuotas

- **Proyectos en estado SUSPENDED hasta tener saldo en AI Cloud** (2026-09). El mensaje no dice por qué. Propuesta: indicar la causa.
- **eu-north1 sin cuota de instancias solo CPU** (2026-09). Todo, hasta un paso de descarga, tiene que correr en una L40S. Por eso el pipeline va en un único trabajo con GPU.
- **Precio spot como "pricing policy"** (2026-10-04). Costó entender que es una puja por capacidad interrumpible (`--preemptible --follows-spot-price`, L40S a mitad de precio). Propuesta: explicarlo con esas palabras en la ayuda de la CLI.

## Nebius Token Factory

- **Ningún Nemotron de Token Factory acepta imágenes** (comprobado 2026-09-28: Nano 30B, 3.5 Lightning, Super 120B, Ultra 550B). Para la parte de visión usamos GLM-5.3-Flash. Propuesta: ofrecer un Nemotron con visión (Nemotron Nano VL o similar).
- **SAM 3, SigLIP 2 y modelos de detección no están en Token Factory** (2026-09). Tenemos que levantarlos en AI Cloud, con el coste de arranque de arriba. Propuesta: endpoints de modelos de visión no generativos (segmentación, embeddings, detección).
- **API compatible con OpenAI y `/v1/models` con IDs y precios** (positivo). Nos permitió elegir modelo por rol y mostrar al usuario el coste real de cada paso.
- **El saldo se queda en negativo sin cortar el servicio y luego llega un cargo a la tarjeta sin aviso previo** (2026-10-12). Con el crédito agotado, el saldo bajó a -0,70 $ y el banco notificó un cargo de 1,20 $ que no esperábamos. La documentación dice que la tarjeta se cobra a principio de mes si el saldo es negativo o al llegar a un umbral de facturación, pero ese umbral lo fija Nebius (región, uso, historial) y no se puede bajar. Los presupuestos de Billing solo avisan por correo, con horas de retraso, y no paran nada. No hay forma documentada de poner un tope duro ni de desactivar el cobro automático con tarjeta (la transferencia es solo para empresas). Para un participante de hackathon o un desarrollador individual es un riesgo: un bucle con un error puede gastar sin límite. Propuesta: un tope de gasto duro por proyecto o por clave de API que corte las peticiones (error 402) al alcanzarse, y un aviso por correo antes del primer cargo cuando se acaba el crédito promocional.

## NVIDIA (Isaac Sim, Replicator, assets)

- **`print` sin `flush=True` se pierde y `app.close()` mata el proceso antes de imprimir la traza de una excepción** (2026-09). Errores silenciosos en modo headless.
- **`BasicWriter` con ruta relativa escribe en `~/omni.replicator_out/`** (2026-09), no en el directorio actual.
- **Los HDRI remotos en una DomeLight cargan de forma asíncrona**: las primeras capturas salen negras o blancas (2026-09). Hay que bajarlos y desactivar el streaming asíncrono por ajustes de carb.
- **Rutas del bucket público de cielos**: las reales son `Assets/Skies/2022_1/Skies/<tipo>/<nombre>.hdr`; una ruta mal escrita devuelve un XML de error de ~300 bytes con código 200 (2026-09).
- **Los assets USD remotos pueden no estar cargados al medirlos**: caja de altura 0, y un árbol en centímetros acaba midiendo kilómetros (2026-09).
- **No hay vehículos civiles modernos en los assets públicos de NVIDIA** (solo carretillas y militares) (2026-09). Para un caso de tráfico tuvimos que ir a Objaverse.
- **Sin RT cores no hay render**: H100/H200 no sirven para Isaac Sim, hay que usar L40S. Bien documentado por NVIDIA, pero conviene que Nebius lo diga al elegir GPU para Isaac.

## Tavily

- (Pendiente: probar con entrevistas reales; ver `TODO.md`, acción `research` de la entrevista.)

## Ideas de producto para ellos

- Plantilla oficial de "Isaac Sim en Serverless Jobs" con imagen fina, cachés de shaders en un bucket y entornos como kits: es lo que hemos tenido que montar a mano.
- **"Sign in with Nebius" para apps de terceros** (2026-10-05). No hay OAuth de Nebius como proveedor de identidad: una app no puede pedir permiso al usuario para gastar de su cuenta, solo pedirle que pegue una API key o la clave de un service account. Propuesta: OAuth con permisos acotados (Token Factory y Serverless Jobs en un proyecto) para que las apps hechas sobre Nebius lleven clientes a Nebius sin manejar sus credenciales.
