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
    if (!blob) return null;
    const f35 = /^image\/jpe?g$/.test(file.type) ? focal35(new Uint8Array(await file.slice(0, 256 * 1024).arrayBuffer())) : null;
    const out = f35 ? withFocal35(new Uint8Array(await blob.arrayBuffer()), f35) : blob;
    return new File([out], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" });
  } finally {
    img.close();
  }
}

// The canvas drops the EXIF, and the scene route needs the 35 mm equivalent focal length (scene_job.no_focal):
// read it from the original and put it back alone, in a minimal EXIF block (no orientation: the pixels are
// already upright).
const FOCAL35 = 0xa405, EXIF_IFD = 0x8769;

// The 35 mm equivalent focal length in a JPEG's EXIF, or null.
export function focal35(jpeg: Uint8Array): number | null {
  const v = new DataView(jpeg.buffer, jpeg.byteOffset, jpeg.byteLength);
  if (v.byteLength < 4 || v.getUint16(0) !== 0xffd8) return null;
  for (let p = 2; p + 4 <= v.byteLength && v.getUint8(p) === 0xff;) {
    const marker = v.getUint8(p + 1), len = v.getUint16(p + 2);
    if (marker === 0xda) break;  // the image data: no EXIF after it
    if (marker === 0xe1 && p + 10 <= v.byteLength && v.getUint32(p + 4) === 0x45786966) {  // "Exif"
      try {
        return exifFocal35(new DataView(jpeg.buffer, jpeg.byteOffset + p + 10, Math.min(len - 8, v.byteLength - p - 10)));
      } catch {
        return null;  // a broken or cut EXIF block
      }
    }
    p += 2 + len;
  }
  return null;
}

function exifFocal35(t: DataView): number | null {
  const le = t.getUint16(0) === 0x4949;
  const entry = (ifd: number, tag: number) => {
    for (let i = 0, n = t.getUint16(ifd, le); i < n; i++) {
      const e = ifd + 2 + 12 * i;
      if (t.getUint16(e, le) === tag) return e;
    }
    return null;
  };
  const sub = entry(t.getUint32(4, le), EXIF_IFD);
  const f = sub === null ? null : entry(t.getUint32(sub + 8, le), FOCAL35);
  const mm = f === null ? 0 : t.getUint16(f + 8, le);
  return mm > 0 ? mm : null;
}

// The JPEG with an EXIF block holding only the 35 mm equivalent focal length, after its JFIF block if any.
export function withFocal35(jpeg: Uint8Array, mm: number): Uint8Array<ArrayBuffer> {
  const tiff = new DataView(new ArrayBuffer(44));  // header, IFD0 with the Exif IFD's offset, the Exif IFD
  tiff.setUint16(0, 0x4949); tiff.setUint16(2, 42, true); tiff.setUint32(4, 8, true);
  tiff.setUint16(8, 1, true);
  tiff.setUint16(10, EXIF_IFD, true); tiff.setUint16(12, 4, true); tiff.setUint32(14, 1, true); tiff.setUint32(18, 26, true);
  tiff.setUint16(26, 1, true);
  tiff.setUint16(28, FOCAL35, true); tiff.setUint16(30, 3, true); tiff.setUint32(32, 1, true); tiff.setUint16(36, mm, true);
  const app1 = new Uint8Array(4 + 6 + 44);
  const h = new DataView(app1.buffer);
  h.setUint16(0, 0xffe1); h.setUint16(2, app1.length - 2);
  app1.set([0x45, 0x78, 0x69, 0x66, 0, 0], 4);  // "Exif\0\0"
  app1.set(new Uint8Array(tiff.buffer), 10);
  const v = new DataView(jpeg.buffer, jpeg.byteOffset, jpeg.byteLength);
  const at = v.getUint16(2) === 0xffe0 ? 4 + v.getUint16(4) : 2;
  const out = new Uint8Array(jpeg.length + app1.length);
  out.set(jpeg.subarray(0, at)); out.set(app1, at); out.set(jpeg.subarray(at), at + app1.length);
  return out;
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
