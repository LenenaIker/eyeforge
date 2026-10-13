// Phone photos are 3-12 MB: shrink them in the browser before uploading, so a mobile connection sends a
// fraction of it. The teacher model and the training never need more than LONG_SIDE pixels.
const LONG_SIDE = 1600;

// The photo ready to upload, or null when this browser cannot read it (e.g. HEIC on some): the caller says so
// instead of sending a 10 MB original without a word. One at a time: 11 phone photos decoded at once (36 MB of
// pixels each) left a Pixel 7a out of memory and every one went up unshrunk (2026-10-13).
export async function shrink(file: File): Promise<File | null> {
  const img = await decode(file);
  if (!img) return null;
  try {
    const scale = Math.min(1, LONG_SIDE / Math.max(img.width, img.height));
    if (scale === 1 && file.size < 1_500_000 && /^image\/(jpeg|png|webp)$/.test(file.type)) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    canvas.getContext("2d")!.drawImage(img.source, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((ok) => canvas.toBlob(ok, "image/jpeg", 0.9));
    canvas.width = canvas.height = 0;  // frees the pixels now, not at the next garbage collection
    return blob && new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" });
  } finally {
    img.close();
  }
}

type Decoded = { source: CanvasImageSource; width: number; height: number; close: () => void };

// createImageBitmap first ("from-image" applies the EXIF orientation, so the result is upright like the photo
// on the phone); an <img> when it fails, which browsers also draw upright.
async function decode(file: File): Promise<Decoded | null> {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    return { source: bmp, width: bmp.width, height: bmp.height, close: () => bmp.close() };
  } catch { /* try the <img> */ }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return { source: img, width: img.naturalWidth, height: img.naturalHeight, close: () => URL.revokeObjectURL(url) };
  } catch {
    URL.revokeObjectURL(url);
    return null;
  }
}

// The key the server gives a photo (workspace.content_key: sha256 of the bytes, 16 hex characters), or null
// where the browser has no crypto.subtle (plain http to a LAN address): then nothing is skipped.
export async function contentKey(file: Blob): Promise<string | null> {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}
