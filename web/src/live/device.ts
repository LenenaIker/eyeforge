// Live testing is for phones and tablets only (user, 2026-09-27): a desktop has no back camera to point at
// things. Touch-first screen or a mobile browser counts; iPadOS reports itself as a Mac, so touch points too.
export function isHandheld(): boolean {
  const coarse = window.matchMedia("(pointer: coarse)").matches && window.matchMedia("(hover: none)").matches;
  const mobileUa = (navigator as any).userAgentData?.mobile === true ||
    /android|iphone|ipad|ipod|mobile/i.test(navigator.userAgent);
  const ipad = /macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1;
  return coarse || mobileUa || ipad;
}

/** Full screen and portrait lock, like a camera app. Browsers only allow it from a user gesture, so it is called
 * from the tap that opens the live test. Where it is refused (iOS, some browsers), the live page still works:
 * it fills the screen anyway and turns the frame by the gravity sensor. */
export async function enterCameraMode() {
  try {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen({ navigationUI: "hide" });
    await (screen.orientation as any).lock?.("portrait-primary");
  } catch { /* not supported here */ }
}
