# Plan: login, créditos y acceso del jurado en la demo pública

## Contexto

La demo pública (Hetzner CX23, decisión del usuario 2026-10-03) necesita una URL que funcione. Hoy la protege una clave compartida (`SIM2CHIP_WEB_KEY`), pero hay 16.766 inscritos en Devpost y una clave en las instrucciones puede filtrarse. El usuario quiere cobrar por ejecutar (AGPL lo permite) y no puede pagar GPU a miles de usuarios.

Reglas del hackathon (https://nebiusglobalaihackathon.devpost.com/rules):
- El proyecto debe estar disponible "free of charge and without any restriction" para el patrocinador, la organización y el jurado hasta el final de la evaluación.
- Si la web es privada, las credenciales van en las "testing instructions".
- Cobrar a otros usuarios no está prohibido.

Decisiones del usuario (2026-10-03):
- Login con Google o GitHub.
- **Cero ejecuciones gratis para un usuario normal.**
- Créditos gratis solo para cuentas con un correo verificado de los dominios del patrocinador y la organización.
- Un aviso para que un juez que entre con otra cuenta contacte con el usuario, que le da créditos a mano.
- Respuestas del 2026-10-03:
  - **Se cobra todo**, entrevista incluida: nada es gratis salvo para cuentas verificadas.
  - **Contacto:** un correo del dominio (Porkbun lo reenvía al personal).
  - **Cuentas de jurado sin límite**, porque las reglas dicen gratis y sin restricción. Si alguien abusa, el usuario lo bloquea a mano con el mensaje de "bloqueo preventivo automático" y pide perdón si era un juez. Al acabar la evaluación, esas cuentas pasan a ser normales.

## Comportamiento

- **Landing:** abierta para todos, sin login.
- **App (`/app`):** pide entrar con Google o GitHub.
- **Niveles de cuenta:**
  - **Jurado:** uso sin límite (entrevista y ejecuciones) hasta `SIM2CHIP_JUDGE_UNTIL` (fin de la evaluación); después pasa a normal. Se obtiene así:
    - automáticamente, si la cuenta trae un correo verificado de `nvidia.com`, `nebius.com`, `tavily.com` o `devpost.com`. Así se cubre a jurado, patrocinador y organización sin conocerlos.
      - Google: claims `email` y `email_verified` del token OIDC.
      - GitHub: `GET /user/emails` con el scope `user:email`, solo los correos `verified: true`.
    - a mano, con `sim2chip accounts judge <correo>`, para un juez que entre con su cuenta personal y escriba al contacto.
  - **Normal:** todo cuesta créditos, también la entrevista. Empieza con 0. La app muestra el saldo y, sin créditos, explica que hace falta comprarlos (el cobro llega después). Unidad: una ejecución completa; la entrevista cuesta una fracción pequeña.
  - **Bloqueada:** `sim2chip accounts ban <correo>`. La app muestra "Your account was paused by an automatic safety check. If this is a mistake, contact <contacto>".
- **Aviso para jueces**, mientras dure la evaluación: "Judging this hackathon? Write to <contacto> from your account's email and we will give it free access". El contacto es `SIM2CHIP_CONTACT`, un correo de `eyeforge.app` reenviado por Porkbun.
- **Mantenimiento en el servidor, por SSH:** `sim2chip accounts list|judge|ban|unban` y `sim2chip credits grant <correo> <n>`.
- **Trabajos por cuenta, no por navegador:** quien entra con la misma cuenta en el móvil ve los trabajos creados en el PC. Resuelve la prueba en vivo, que solo existe en el móvil.
- **Tope diario de gasto** (`SIM2CHIP_DAILY_USD`, ya existe): se mantiene como red de seguridad global.
- **Cobro** (Stripe u otro): fuera de este plan. Los créditos son la base sobre la que se conectará.

## Diseño técnico

- **OAuth** en el servidor FastAPI. Rutas:
  - `/auth/google/login` y `/auth/google/callback`;
  - `/auth/github/login` y `/auth/github/callback`;
  - `POST /auth/logout`.
  - Flujo authorization code con `state` contra CSRF. Con `authlib` o a mano con `httpx`, sin servicios externos.
- **Sesión:** cookie firmada (`itsdangerous`, secreto `SIM2CHIP_SESSION_SECRET`), `httponly`, `secure`, `samesite=lax` (necesario para volver del proveedor). Guarda el id de cuenta.
- **Cuentas y créditos:** SQLite en `SIM2CHIP_HOME/accounts.db`.
  - Tabla `accounts` (id, proveedor, id del proveedor, correo, correo verificado, creada).
  - Columnas de nivel en `accounts`: `judge` (dominio o a mano) y `banned`.
  - Tabla `credits` (movimientos con motivo: `grant`, `purchase`, `run`, `interview`, `refund`).
  - El saldo es la suma de movimientos.
  - Una ejecución descuenta al lanzarse y devuelve el crédito si falla antes de usar la GPU.
- **Propietario de trabajos:** el fichero `owner` guarda el id de cuenta en vez del visitante. Los trabajos de visitantes anteriores ya no se verán, lo cual da igual porque aún no hay demo pública.
- **API:**
  - `/api/me`: cuenta, saldo y si es jurado.
  - `/api/config` sigue abierto.
  - Todo lo demás de `/api` pide sesión.
  - Empezar una entrevista y lanzar un trabajo (`POST /api/interview`, `POST /api/jobs`, `train`, `photos`, `review`) comprueban el nivel o el saldo; una cuenta bloqueada recibe 403 en todo.
- **Clave compartida:** sigue sirviendo para el modo local o privado (`--public` sin OAuth configurado). Con OAuth configurado, la clave deja de usarse.
- **Web:**
  - Pantalla de entrada con "Continue with Google" y "Continue with GitHub", en el lugar del formulario de clave de `AppShell`.
  - Saldo en la cabecera.
  - El aviso para jueces.
  - El botón de entrenar desactivado sin créditos, con la explicación.

## Lo que tiene que hacer el usuario

1. **Google Cloud:** crear un cliente OAuth "Web application" con la URI de redirección `https://eyeforge.app/auth/google/callback` (y `http://localhost:8000/...` para probar). Pantalla de consentimiento en modo producción, con los scopes básicos `openid email profile`, que no piden verificación de Google.
2. **GitHub:** crear una OAuth App con el callback `https://eyeforge.app/auth/github/callback`.
3. Crear en Porkbun el correo de contacto de `eyeforge.app` con reenvío al personal.
4. Poner en `.env`: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `SIM2CHIP_SESSION_SECRET`, `SIM2CHIP_CONTACT` y `SIM2CHIP_JUDGE_UNTIL`. Nunca por chat.
5. Crear el servidor CX23 en Hetzner y apuntar el DNS de `eyeforge.app` en Porkbun.

## Verificación

1. Tests del servidor con un proveedor OAuth falso:
   - login y callback;
   - nivel de jurado por dominio verificado, y ninguno si el correo no está verificado;
   - una cuenta normal sin saldo no puede entrevistar ni entrenar;
   - el nivel de jurado caduca en `SIM2CHIP_JUDGE_UNTIL`;
   - una cuenta bloqueada recibe 403;
   - devolución si la ejecución falla;
   - trabajos por cuenta;
   - `accounts judge|ban` y `credits grant`.
2. Prueba real en local con los dos proveedores.
3. En el servidor: entrar con una cuenta normal (sin créditos) y con una de jurado (con `accounts judge`), lanzar una ejecución en Nebius y verla desde el móvil con la misma cuenta.
