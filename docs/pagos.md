# Pagos con Stripe

Recargas del saldo (en dólares) desde Billing y recarga automática. Código en `sim2chip/payments.py`. Se cobra por uso: coste real ×2 (`sim2chip/accounts.py`, decisión del usuario 2026-10-09).

## Cómo funciona

- **Recarga:** el usuario elige 5, 10, 25 o 50 $ y va a la página de pago de Stripe (Checkout). Los datos de la tarjeta nunca pasan por nuestro servidor.
- **Cuándo entra el dinero:** solo cuando Stripe lo confirma por el webhook firmado. Cada pago cuenta una vez, aunque Stripe lo avise dos veces.
- **Tarjeta guardada:** la del primer pago queda en Stripe y Billing la muestra como "visa 4242".
- **Recarga automática:** "si el saldo baja de X $, añade Y $", con la tarjeta guardada.
  - Como mucho una cada 10 minutos.
  - Si la tarjeta falla o pide 3-D Secure, se apaga sola y Billing dice por qué.
- **Devoluciones:** al devolver un pago desde el panel de Stripe, el saldo baja lo devuelto (aunque quede en negativo si ya se gastó).

## Alta (lo hace el usuario)

1. Crear la cuenta en https://dashboard.stripe.com. Para cobrar de verdad, Stripe pide los datos fiscales: autónomo o empresa, y una cuenta bancaria.
2. Modo de pruebas: en Developers > API keys, copiar la `sk_test_...` a `.env` como `STRIPE_SECRET_KEY`.
3. Webhook: en Developers > Webhooks, añadir el endpoint `https://eyeforge.app/api/stripe/webhook` con los eventos `checkout.session.completed`, `payment_intent.succeeded` y `charge.refunded`. Copiar su `whsec_...` a `.env` como `STRIPE_WEBHOOK_SECRET`.
4. Reiniciar el servidor: Billing ya enseña las recargas.
5. Probar con la tarjeta `4242 4242 4242 4242` (cualquier fecha futura y CVC). Con `4000 0000 0000 9995` se rechaza.
6. Para producción, repetir los pasos 2 y 3 con las claves `sk_live_` en modo live.

**Probar en local** sin dominio: con la CLI de Stripe, `stripe listen --forward-to localhost:8011/api/stripe/webhook`. Imprime un `whsec_` temporal para `.env`.

## IVA

- Vendiendo a particulares de la UE, el IVA del país del comprador se declara por la ventanilla única (OSS).
- `SIM2CHIP_STRIPE_TAX=1` hace que Stripe Tax lo calcule y lo añada en la página de pago. Antes hay que activarlo y registrar el origen en el panel de Stripe. Cuesta un 0,5% por venta.
- Con IVA activado la recarga automática no se ofrece todavía: ese cobro se hace sin página de pago y se saltaría el cálculo. Pendiente: la Tax Calculation API.
- Consultarlo con una gestoría antes de pasar a live.

## Pendiente

- Probar con claves reales de test (no hay ninguna en `.env` todavía).
- Recibos: Stripe los envía por correo si se activa en Settings > Emails.
