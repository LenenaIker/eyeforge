// The card picture of a 3D model the user uploads, rendered in the browser (the server has no GPU to spare): the
// model on white, seen from the front and a little above, filling the frame. three.js loads only when needed.

export async function glbCard(file: File, size = 512): Promise<Blob | null> {
  const THREE = await import("three");
  const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
  const gltf = await new GLTFLoader().parseAsync(await file.arrayBuffer(), "");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#ffffff");
  scene.add(gltf.scene);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f94, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(2, 4, 3);
  scene.add(sun);

  const box = new THREE.Box3().setFromObject(gltf.scene);
  if (box.isEmpty()) return null;
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getBoundingSphere(new THREE.Sphere()).radius;
  const camera = new THREE.PerspectiveCamera(35, 1, radius / 100, radius * 100);
  const dist = radius / Math.sin(THREE.MathUtils.degToRad(35 / 2)) * 1.05;
  const dir = new THREE.Vector3(0.6, 0.45, 1).normalize();
  camera.position.copy(center).addScaledVector(dir, dist);
  camera.lookAt(center);

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  try {
    renderer.setSize(size, size, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.render(scene, camera);
    return await new Promise((done) => renderer.domElement.toBlob(done, "image/png"));
  } finally {
    renderer.dispose();
  }
}
