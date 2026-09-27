// Phone photos are 3-12 MB: shrink them in the browser before uploading, so a mobile connection sends a
// fraction of it. The teacher model and the training never need more than LONG_SIDE pixels.
const LONG_SIDE = 1600;

export async function prepareImages(files: File[]): Promise<File[]> {
  return Promise.all(files.map(shrink));
}

async function shrink(file: File): Promise<File> {
  try {
    // "from-image" applies the EXIF orientation, so the result is upright like the photo on the phone.
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, LONG_SIDE / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size < 1_500_000) { bmp.close(); return file; }
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    bmp.close();
    const blob = await new Promise<Blob | null>((ok) => canvas.toBlob(ok, "image/jpeg", 0.9));
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" });
  } catch {
    return file;  // a format the browser cannot decode (e.g. HEIC on some): send it as it is
  }
}
