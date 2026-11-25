
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const D = window.PAGE_DATA || {};
const VER = encodeURIComponent(D.version || '1');

const COND = {
  'Knight': { kind: 'Mesh + text', img: 'static/images/app_cond/knight_mesh.jpg' },
  'Penguin girl': { kind: 'Image + text', img: 'static/images/app_cond/penguin_photo.jpg' },
  'Calico cat + mug': { kind: 'Text' },
  'Jump shot': { kind: 'Image + text', img: 'static/images/app_cond/jumpshot_photo.jpg' },
  'Red car hits blue car': { kind: 'Image + text', img: 'static/images/app_cond/cars_photo.jpg' },
  'Plush boxer': { kind: 'Image + text', img: 'static/images/app_cond/plush_boxer_photo.jpg' },
  'Robot': { kind: 'Mesh + text', img: 'static/images/app_cond/robot_mesh.jpg' },
  'Ceiling fan': { kind: 'Mesh + text', img: 'static/images/app_cond/fan_mesh.jpg' },
  'Enamel pot': { kind: 'Mesh + text', img: 'static/images/app_cond/pot_mesh.jpg' },
  'Castle door': { kind: 'Mesh + text', img: 'static/images/app_cond/door_mesh.jpg' },
  'Paper crane': { kind: 'Image + text', img: 'static/images/app_cond/paper_crane_photo.jpg' },
  'Monstera plant': { kind: 'Image + text', img: 'static/images/app_cond/plant_photo.jpg' },
  'Astronaut on a horse': { kind: 'Image + text', img: 'static/images/app_cond/horse_astro_photo.jpg' },
  'Flag': { kind: 'Image + text', img: 'static/images/app_cond/flag_photo.jpg' },
  'Robot spider': { kind: 'Image + text', img: 'static/images/app_cond/spider_photo.jpg' },
  'Desk lamp': { kind: 'Video' }, 'Whale-hood girl': { kind: 'Video' }, 'Wipes canister': { kind: 'Video' },
  'Sunglasses': { kind: 'Video' }, 'Crocodile with milk + gift': { kind: 'Video' }, 'Mannequin dancer': { kind: 'Video' },
  'Whale-sweater girl': { kind: 'Video' }, 'Waffle maker': { kind: 'Video' },
  "Newton's cradle": { kind: 'Video' },
  'Basketball dancer': { kind: 'Video' }
};

const PICK = [
  ['app114', 'The girl twists, raises her flippers, bends and points'],
  ['app002', 'The knight raises his sword and strikes down'],
  ['app481', 'A mannequin dances'],
  ['app468', 'A man dribbles a basketball'],
  ['cb931', "The balls of a Newton's cradle swing and click"],
  ['app326', 'The robot does a slow backflip'],
  ['app241', 'The fan blades slowly spin'],
  ['app233', 'The lid lifts off the pot'],
  ['app261', 'Both doors swing open'],
  ['app201', 'A cat taps a mug and tips it over'],
  ['app471', 'One temple arm of the sunglasses folds in'],
  ['app601', 'The red car drives into the blue car'],
  ['app181', 'The horse walks forward as the astronaut waves'],
  ['app304', 'A corgi wags its tail and turns its head'],
  ['app473', 'The lid of a waffle maker closes'],
  ['app962', 'A paper crane flaps its wings'],
  ['app493', 'A girl in a whale sweater sways and wags her tail'],
  ['app907', 'The spider steps in place and turns to its side'],
  ['app450', 'A desk lamp turns its head toward the camera'],
  ['app305', 'A rabbit sits up and twitches its ears'],
  ['app373', 'A goldfish swishes its tail'],
  ['app971', 'A flag waves in the wind'],
  ['app384', 'The monstera leaves sway in a breeze'],
  ['app472', 'The lid of a wipes canister closes'],
  ['app490', 'A crocodile walks toward the camera carrying two boxes'],
  ['app301', 'A dog sits, then lifts a front paw to shake hands'],
  ['app282', 'A robot arm picks up the red cube and places it on the green mat'],
  ['app352', 'A yellow flower slowly closes into a bud'],
  ['app531', 'A player releases a jump shot'],
  ['app707', 'A plush toy does a hip-hop dance in place'],
];
const Q = new URLSearchParams(location.search);
const KINDS = ['Text', 'Image + text', 'Mesh + text', 'Video'];
const kindCls = (k) => 'app3-kind--' + k.toLowerCase().replace(/[^a-z]+/g, '-');

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

async function gunzip(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' ' + r.status);
  const ds = r.body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(ds).arrayBuffer());
}
const cache = new Map();
function loadClip(name) {
  if (cache.has(name)) return cache.get(name);
  const base = `static/mesh3d/${name}/`;
  const p = (async () => {
    const meta = await (await fetch(base + 'meta.json?v=' + VER)).json();
    const [topo, anim] = await Promise.all([gunzip(base + 'topo.bin.gz?v=' + VER), gunzip(base + 'anim.bin.gz?v=' + VER)]);
    const F = meta.faces, R = meta.render, N = meta.base, T = meta.frames;
    const tb = topo.buffer.slice(topo.byteOffset, topo.byteOffset + topo.byteLength);
    let o = 0;
    const index = new Uint32Array(tb, o, F * 3); o += F * 12;
    const uv = new Float32Array(tb, o, R * 2); o += R * 8;
    const baseOf = new Uint32Array(tb, o, R); o += R * 4;
    const smooth = new Uint32Array(tb, o, R);

    const M = T * 3 * N, q = new Int32Array(M);
    for (let j = 0; j < M; j++) { const z = anim[j] | (anim[M + j] << 8); q[j] = (z >>> 1) ^ -(z & 1); }
    for (let r = 0; r < T * 3; r++) { const s = r * N; for (let i = 1; i < N; i++) q[s + i] += q[s + i - 1]; }
    const S = 3 * N;
    for (let t = 1; t < T; t++) {
      const a = t * S, b = a - S, c = b - S;
      if (t === 1) for (let j = 0; j < S; j++) q[a + j] += q[b + j];
      else for (let j = 0; j < S; j++) q[a + j] += 2 * q[b + j] - q[c + j];
    }
    const pos = new Float32Array(M), lo = meta.lo, st = meta.step;
    for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) {
      const src = (t * 3 + k) * N, dst = t * S;
      for (let i = 0; i < N; i++) pos[dst + i * 3 + k] = lo[k] + q[src + i] * st;
    }

    const box = new THREE.Box3(), v = new THREE.Vector3();
    for (let j = 0; j < M; j += 3) { v.set(pos[j], pos[j + 1], pos[j + 2]); box.expandByPoint(v); }
    const cx = (box.min.x + box.max.x) / 2, cz = (box.min.z + box.max.z) / 2;
    let r2 = 0;
    for (let j = 0; j < M; j += 3) { const dx = pos[j] - cx, dz = pos[j + 2] - cz, d = dx * dx + dz * dz; if (d > r2) r2 = d; }
    return { name, meta, index, uv, baseOf, smooth, pos, box, radius: Math.sqrt(r2) };
  })();
  cache.set(name, p);
  p.catch(() => cache.delete(name));
  return p;
}

const NEUTRAL = new THREE.Color('#c3c8cd');
const texLoader = new THREE.TextureLoader();

class Viewer {
  constructor(host) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0xffffff, 1);
    host.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
    this.scene.add(this.camera);
    this.scene.add(new THREE.HemisphereLight(0xffffff, new THREE.Color('#c0c8d0'), 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.9); key.position.set(-1.2, 2.2, 2.4); this.camera.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.55); fill.position.set(2.5, 0.6, 1); this.camera.add(fill);
    key.target.position.set(0, 0, -1); fill.target.position.set(0, 0, -1); this.camera.add(key.target, fill.target);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, enablePan: false, autoRotate: true, autoRotateSpeed: 1.6, rotateSpeed: 0.8 });
    this.controls.addEventListener('start', () => { this.touched = true; this.controls.autoRotate = false; clearTimeout(this._resume); this.onInteract && this.onInteract(); });
    this.controls.addEventListener('end', () => { clearTimeout(this._resume); this._resume = setTimeout(() => { this.controls.autoRotate = this.spin; }, 6000); });
    this.spin = true;

    this.zoomGoal = 0;
    host.addEventListener('wheel', (e) => {
      if (!this.clip) return;
      e.preventDefault(); e.stopPropagation();
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
      const cur = this.zoomGoal || this.camera.position.distanceTo(this.controls.target);
      this.zoomGoal = THREE.MathUtils.clamp(cur * Math.exp(dy * 0.0011), this.controls.minDistance, this.controls.maxDistance);
      this.touched = true; this.anim = null; this.onInteract && this.onInteract();
    }, { capture: true, passive: false });
    this.controls.addEventListener('start', () => { this.zoomGoal = 0; this.anim = null; });

    const cv = document.createElement('canvas'); cv.width = cv.height = 128;
    const g = cv.getContext('2d'), grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grd.addColorStop(0, 'rgba(70,50,40,0.26)'); grd.addColorStop(0.5, 'rgba(70,50,40,0.09)'); grd.addColorStop(1, 'rgba(70,50,40,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
    this.shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, depthWrite: false }));
    this.shadow.rotation.x = -Math.PI / 2; this.shadow.renderOrder = -1; this.scene.add(this.shadow);
    this.gray = false; this.frame = -1; this.clip = null; this.visible = true;
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    const loop = () => { requestAnimationFrame(loop); if (this.visible) this.tick(); };
    requestAnimationFrame(loop);
  }
  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    if (!this.touched) this.fit();
  }
  setClip(c) {
    if (this.mesh) { this.scene.remove(this.mesh, this.wire); this.mesh.geometry.dispose(); this.mats.forEach((m) => { m.map && m.map.dispose(); m.dispose(); }); }
    this.clip = c; this.frame = -1; this.touched = false;
    const { meta, index, uv, baseOf, smooth } = c, R = meta.render, F = meta.faces;
    const geo = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(new Float32Array(R * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.nrmAttr = new THREE.BufferAttribute(new Float32Array(R * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', this.posAttr); geo.setAttribute('normal', this.nrmAttr);
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    meta.groups.forEach((gr) => geo.addGroup(gr.start, gr.count, gr.materialIndex));

    this.cb = new Uint32Array(F * 3); this.cs = new Uint32Array(F * 3);
    for (let j = 0; j < F * 3; j++) { this.cb[j] = baseOf[index[j]]; this.cs[j] = smooth[index[j]]; }
    this.sn = new Float32Array(meta.smooth * 3);
    this.texMats = meta.materials.map((m) => {
      const mat = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
      mat.color.setRGB(m.color[0], m.color[1], m.color[2]);
      if (m.map) {
        const t = texLoader.load(`static/mesh3d/${c.name}/${m.map}?v=${VER}`);
        t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4;
        mat.map = t;
      }
      return mat;
    });
    this.grayMat = new THREE.MeshLambertMaterial({ color: NEUTRAL, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    this.mats = this.texMats.concat([this.grayMat]);
    this.mesh = new THREE.Mesh(geo, this.gray ? this.grayMat : this.texMats);
    this.wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x4a4f57, wireframe: true, transparent: true, opacity: 0.16, depthWrite: false }));
    this.mats.push(this.wire.material);
    this.mesh.frustumCulled = this.wire.frustumCulled = false;
    this.wire.visible = this.gray;
    this.scene.add(this.mesh, this.wire);

    this.fit();
    const ext = Math.max(c.box.max.x - c.box.min.x, c.box.max.z - c.box.min.z);
    this.shadow.scale.set(ext * 1.15, ext * 1.15, 1);
    this.shadow.position.set((c.box.min.x + c.box.max.x) / 2, c.box.min.y - 0.002 * ext, (c.box.min.z + c.box.max.z) / 2);
    this.controls.autoRotate = this.spin;
    this.showFrame(0);
  }
  fit(animate) {
    const c = this.clip; if (!c) return;
    const ctr = c.box.getCenter(new THREE.Vector3());
    const dir = new THREE.Vector3().fromArray(c.meta.camera).sub(ctr); dir.y = Math.max(dir.y, 0.15 * dir.length()); dir.normalize();

    const m = 0.9, tv = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2) * m, th = tv * this.camera.aspect;
    const r = c.radius, y0 = c.box.min.y - ctr.y, y1 = c.box.max.y - ctr.y;
    const up0 = new THREE.Vector3(0, 1, 0), right = new THREE.Vector3().crossVectors(up0, dir).normalize(), up = new THREE.Vector3().crossVectors(dir, right);
    const pt = new THREE.Vector3();
    let dist = 0;
    for (let k = 0; k < 32; k++) {
      const a = k / 32 * Math.PI * 2;
      for (const y of [y0, y1]) {
        pt.set(r * Math.cos(a), y, r * Math.sin(a));
        const x = pt.dot(right), yy = pt.dot(up), z = pt.dot(dir);
        dist = Math.max(dist, z + Math.abs(x) / th, z + Math.abs(yy) / tv);
      }
    }
    this.controls.minDistance = dist * 0.3; this.controls.maxDistance = dist * 2.5;
    this.zoomGoal = 0;
    const goal = ctr.clone().addScaledVector(dir, dist);
    if (animate) {
      this.anim = { t0: performance.now(), p0: this.camera.position.clone(), q0: this.controls.target.clone(), p1: goal, q1: ctr };
      return;
    }
    this.anim = null;
    this.controls.target.copy(ctr);
    this.camera.position.copy(goal);
    this.controls.update();
  }
  reset() {
    this.touched = false; clearTimeout(this._resume);
    this.fit(true);
    this.controls.autoRotate = this.spin;
  }
  setGray(on) {
    this.gray = on;
    if (!this.mesh) return;
    this.mesh.material = on ? this.grayMat : this.texMats; this.wire.visible = on;
  }
  setSpin(on) { this.spin = on; this.controls.autoRotate = on; }
  showFrame(t) {
    const c = this.clip; if (!c) return;
    t = Math.max(0, Math.min(c.meta.frames - 1, t | 0));
    if (t === this.frame) return;
    this.frame = t;
    const N = c.meta.base, R = c.meta.render, P = c.pos.subarray(t * N * 3, (t + 1) * N * 3);
    const pos = this.posAttr.array, nrm = this.nrmAttr.array, sn = this.sn, cb = this.cb, cs = this.cs, bo = c.baseOf, sm = c.smooth;
    for (let r = 0; r < R; r++) { const b = bo[r] * 3, d = r * 3; pos[d] = P[b]; pos[d + 1] = P[b + 1]; pos[d + 2] = P[b + 2]; }
    sn.fill(0);
    for (let j = 0; j < cb.length; j += 3) {
      const a = cb[j] * 3, b = cb[j + 1] * 3, e = cb[j + 2] * 3;
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
      const vx = P[e] - P[a], vy = P[e + 1] - P[a + 1], vz = P[e + 2] - P[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (let k = 0; k < 3; k++) { const s = cs[j + k] * 3; sn[s] += nx; sn[s + 1] += ny; sn[s + 2] += nz; }
    }
    for (let r = 0; r < R; r++) {
      const s = sm[r] * 3, d = r * 3, x = sn[s], y = sn[s + 1], z = sn[s + 2], l = Math.hypot(x, y, z) || 1;
      nrm[d] = x / l; nrm[d + 1] = y / l; nrm[d + 2] = z / l;
    }
    this.posAttr.needsUpdate = true; this.nrmAttr.needsUpdate = true;
  }
  tick() {
    if (this.clock) this.showFrame(this.clock());
    const tg = this.controls.target, cam = this.camera.position;
    if (this.anim) {
      const a = this.anim, u = Math.min(1, (performance.now() - a.t0) / 650), k = u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      tg.lerpVectors(a.q0, a.q1, k);

      const d0 = a.p0.clone().sub(a.q0), d1 = a.p1.clone().sub(a.q1), l0 = d0.length(), l1 = d1.length();
      const dir = d0.normalize().lerp(d1.normalize(), k).normalize();
      cam.copy(tg).addScaledVector(dir, l0 + (l1 - l0) * k);
      if (u >= 1) this.anim = null;
    } else if (this.zoomGoal) {
      const off = cam.clone().sub(tg), d = off.length(), nd = d + (this.zoomGoal - d) * 0.14;
      cam.copy(tg).addScaledVector(off.normalize(), nd);
      if (Math.abs(nd - this.zoomGoal) < 1e-4 * this.zoomGoal) this.zoomGoal = 0;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

const SKETCH = '<svg viewBox="9 11 48 48" aria-hidden="true"><g fill="none" stroke="#4a4038" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
  + '<path d="M14 25 H39 V50 H14 Z"/><path d="M14 25 L26 13 H51 L39 25"/><path d="M51 13 V38 L39 50"/></g></svg>';
function pillThumb(label) {
  const t = el('span', 'app-pill-thumb app-pill-thumb--' + label.toLowerCase());
  if (label === 'Text') t.appendChild(el('span', 'app-pill-word', '\u201ca cube\u201d'));
  else if (label === 'Image') t.innerHTML = SKETCH;
  else if (label === 'Mesh') { const i = el('img'); i.src = 'static/images/app_cube/mesh.png?v=' + VER; i.alt = ''; t.appendChild(i); }
  else { const i = el('img'); i.src = 'static/images/app_cube/cube.webp?v=' + VER; i.alt = ''; t.appendChild(i); }
  return t;
}

function build() {
  const host = document.getElementById('app3');
  if (!host || !D.text4d) return;
  const byName = {};
  D.text4d.forEach((grp) => grp.items.forEach((a) => { byName[a.name] = { subject: grp.subject, base: a.base }; }));
  const items = PICK.filter((p) => byName[p[0]]).map(([name, cap]) => {
    const b = byName[name], c = COND[b.subject] || { kind: 'Text' };
    return { name, cap, subject: b.subject, base: b.base, kind: c.kind, img: c.img };
  });

  const hero = el('div', 'app-hero');
  const ins = el('div', 'app-hero-in'); ins.setAttribute('role', 'group'); ins.setAttribute('aria-label', 'Filter by input');
  const PILL = [['Text', 'Text'], ['Image', 'Image + text'], ['Mesh', 'Mesh + text'], ['Video', 'Video']];
  const pills = [];

  const toApps = () => document.getElementById('text4d').scrollIntoView({ behavior: 'smooth', block: 'start' });
  const setFilter = (k, btn) => {
    pills.concat([out]).forEach((x) => x.setAttribute('aria-pressed', x === btn ? 'true' : 'false'));
    thumbs.forEach((t) => { t.hidden = !(k === 'All' || t._it.kind === k); });
    const cur = thumbs.find((t) => t._it.name === current);
    if (cur && cur.hidden) { const first = thumbs.find((t) => !t.hidden); first && select(first._it); }
  };
  PILL.forEach(([label, k]) => {
    const n = items.filter((it) => it.kind === k).length;
    if (!n) return;
    const b = el('button', 'app-pill ' + kindCls(k)); b.type = 'button'; b.title = k + ' \u2192 4D';
    b.append(pillThumb(label), el('span', 'app-pill-label', label));
    b.setAttribute('aria-pressed', 'false');
    b.onclick = () => { setFilter(b.getAttribute('aria-pressed') === 'true' ? 'All' : k, b.getAttribute('aria-pressed') === 'true' ? out : b); toApps(); };
    pills.push(b); ins.appendChild(b);
  });
  const NS = 'http://www.w3.org/2000/svg';
  const wires = document.createElementNS(NS, 'svg'); wires.setAttribute('class', 'app-hero-wires');
  wires.setAttribute('viewBox', '0 0 120 100'); wires.setAttribute('preserveAspectRatio', 'none'); wires.setAttribute('aria-hidden', 'true');
  pills.forEach((p, i) => {
    const y = (i + 0.5) * 100 / pills.length;
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', `M0 ${y} C 55 ${y}, 50 50, 112 50`);
    path.setAttribute('class', 'app-wire ' + p.className.split(' ')[1]);
    wires.appendChild(path);
  });
  const head = document.createElementNS(NS, 'path'); head.setAttribute('d', 'M106 44 L118 50 L106 56'); head.setAttribute('class', 'app-wire-head');
  wires.appendChild(head);
  const out = el('button', 'app-hero-out'); out.type = 'button'; out.title = 'Show all examples';
  out.setAttribute('aria-pressed', 'true');
  const cube = el('span', 'app-cube'); cube.setAttribute('aria-hidden', 'true');
  ['f', 'b', 'l', 'r', 't', 'd'].forEach((f) => cube.appendChild(el('i', 'app-cube-' + f)));
  const cubeWrap = el('span', 'app-cube-wrap'); cubeWrap.append(cube, el('b', 'app-cube-label', '4D'));
  out.append(cubeWrap, el('span', 'app-hero-out-cap', '360\u00b0 viewer'));
  out.onclick = () => { setFilter('All', out); toApps(); };
  hero.append(ins, wires, out);
  const tag = el('p', 'app-hero-tag');
  tag.append(el('span', 'app-hero-star', '\u2726'), document.createTextNode('Any mesh. No template. No rig.'), el('span', 'app-hero-star', '\u2726'));

  const stage = el('div', 'app3-stage');
  const side = el('div', 'app3-side');
  const sideHead = el('div', 'app3-side-head');
  const kindTag = el('span', 'app3-kind'), title = el('span', 'app3-title');
  sideHead.append(kindTag, title);
  const condRow = el('div', 'app3-cond');
  const condImgWrap = el('figure', 'app3-cond-img'), condImg = el('img'), condImgCap = el('figcaption');
  condImg.alt = ''; condImgWrap.append(condImg, condImgCap);
  const prompt = el('p', 'app3-prompt');
  condRow.append(condImgWrap, prompt);
  const vidWrap = el('figure', 'app3-video'), video = el('video'), vidCap = el('figcaption');
  video.muted = true; video.loop = true; video.playsInline = true; video.setAttribute('muted', ''); video.setAttribute('playsinline', ''); video.preload = 'auto';
  vidWrap.append(video, vidCap);
  side.append(sideHead, condRow, vidWrap);

  const view = el('div', 'app3-view');
  const canvasHost = el('div', 'app3-canvas');
  const oursTag = el('span', 'app3-ours', 'Ours · 4D mesh');
  const toggles = el('div', 'app3-toggles');
  const segTex = el('button', null, 'Textured'), segGray = el('button', null, 'Gray + wireframe');
  [segTex, segGray].forEach((b) => { b.type = 'button'; });
  const seg = el('div', 'app3-seg'); seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', 'Shading');
  seg.append(segTex, segGray);
  const spinBtn = el('button', 'app3-spin'); spinBtn.type = 'button'; spinBtn.title = 'Auto-rotate';
  spinBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5c-4.4 0-8 1.8-8 4s3.6 4 8 4 8-1.8 8-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M17 6.5 20 9l-3.2 1.6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 13v6m-2.5-2.5L12 19l2.5-2.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg><span>Spin</span>';
  const resetBtn = el('button', 'app3-spin app3-reset'); resetBtn.type = 'button'; resetBtn.title = 'Reset view';
  resetBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M4.5 4.5v4h4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg><span>Reset</span>';
  resetBtn.onclick = () => { viewer && viewer.reset(); };
  toggles.append(seg, resetBtn, spinBtn);
  const hint = el('div', 'app3-hint', 'Drag to rotate · scroll to zoom');
  const status = el('div', 'app3-status');
  const bar = el('div', 'app3-bar');
  const playBtn = el('button', 'app3-play'); playBtn.type = 'button';
  const range = el('input', 'app3-range'); range.type = 'range'; range.min = 0; range.max = 1; range.step = 1; range.value = 0;
  range.setAttribute('aria-label', 'Frame');
  const clockTxt = el('span', 'app3-clock', '');
  bar.append(playBtn, range, clockTxt);
  view.append(canvasHost, oursTag, toggles, hint, status, bar);
  stage.append(side, view);

  const wall = el('div', 'app3-wall');
  const thumbs = items.map((it) => {
    const t = el('button', 'app3-thumb'); t.type = 'button'; t._it = it;
    t.setAttribute('aria-label', it.subject + ': ' + it.cap);
    t.title = it.cap;
    const box = el('span', 'app3-thumb-box');
    const im = el('img'); im.src = `static/images/app_thumb/${it.name}.jpg?v=${VER}`; im.alt = ''; im.loading = 'lazy';
    const hv = el('video'); hv.muted = true; hv.loop = true; hv.playsInline = true; hv.preload = 'none'; hv.setAttribute('muted', ''); hv.setAttribute('playsinline', '');
    box.append(im, hv, el('span', 'app3-dot ' + kindCls(it.kind)));
    t.append(box, el('span', 'app3-thumb-cap', it.subject));
    t.onmouseenter = () => { if (!hv.src) hv.src = it.base + 'ours_input.mp4?v=' + VER; hv.play().then(() => t.classList.add('is-hover')).catch(() => {}); loadClip(it.name).catch(() => {}); };
    t.onmouseleave = () => { hv.pause(); t.classList.remove('is-hover'); };
    t.onclick = () => { select(it); stage.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); };
    wall.appendChild(t);
    return t;
  });
  const legend = el('div', 'app3-legend');
  KINDS.forEach((k) => { const s = el('span'); s.append(el('i', 'app3-dot ' + kindCls(k)), document.createTextNode(k)); legend.appendChild(s); });

  const mount = document.getElementById('app-hero-mount');
  const h1 = document.querySelector('.site-hero .publication-title'), four = document.getElementById('app-4d');
  if (mount && h1 && four) {

    const fuse = el('div', 'app-fuse'), right = el('div', 'app-fuse-title');
    mount.className = 'app-hero-block'; mount.appendChild(fuse);
    right.append(h1, tag);
    fuse.append(ins, wires, right);
    out.className = 'app-4d-btn'; out.replaceChildren(cube, el('span', 'app-4d-txt', '4D'));
    four.replaceChildren(out);
    const brand = h1.querySelector('.site-brand');
    const layoutWires = () => {
      const r = wires.getBoundingClientRect();
      if (!r.height || getComputedStyle(wires).display === 'none') return;
      const yOf = (e) => { const b = e.getBoundingClientRect(); return ((b.top + b.height / 2) - r.top) / r.height * 100; };
      const ty = yOf(brand);
      wires.querySelectorAll('.app-wire').forEach((path, i) => {
        const y = yOf(pills[i]);
        path.setAttribute('d', `M2 ${y} C 50 ${y}, 45 ${ty}, 104 ${ty}`);
      });
      head.setAttribute('d', `M104 ${ty - 6} L116 ${ty} L104 ${ty + 6}`);
    };
    layoutWires(); window.addEventListener('resize', layoutWires);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(layoutWires);
    setTimeout(layoutWires, 300);
    host.append(stage, wall, legend);
  } else if (mount) { mount.className = 'app-hero-block'; mount.append(hero, tag); host.append(stage, wall, legend); }
  else host.append(hero, tag, stage, wall, legend);

  let viewer = null, current = null, token = 0, playing = true, scrubbing = false;
  const fps = () => (viewer && viewer.clip ? viewer.clip.meta.fps : 24);
  const frames = () => (viewer && viewer.clip ? viewer.clip.meta.frames : 1);
  const frameOf = () => Math.min(frames() - 1, Math.floor(video.currentTime * fps() + 1e-3));
  function setPlaying(p) {
    playing = p;
    playBtn.innerHTML = p ? '<svg viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
      : '<svg viewBox="0 0 24 24"><path d="M7 5v14l12-7z"/></svg>';
    playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
    if (p && inView) video.play().catch(() => {}); else video.pause();
  }
  playBtn.onclick = () => setPlaying(!playing);
  range.oninput = () => { scrubbing = true; video.pause(); video.currentTime = (+range.value + 0.5) / fps(); };
  range.onchange = () => { scrubbing = false; if (playing && inView) video.play().catch(() => {}); };
  function setGray(on) {
    segTex.setAttribute('aria-pressed', on ? 'false' : 'true'); segGray.setAttribute('aria-pressed', on ? 'true' : 'false');
    viewer && viewer.setGray(on);
  }
  segTex.onclick = () => setGray(false); segGray.onclick = () => setGray(true);
  spinBtn.setAttribute('aria-pressed', 'true');
  spinBtn.onclick = () => { const on = spinBtn.getAttribute('aria-pressed') !== 'true'; spinBtn.setAttribute('aria-pressed', on ? 'true' : 'false'); viewer && viewer.setSpin(on); };

  function fillSide(it) {
    kindTag.className = 'app3-kind ' + kindCls(it.kind); kindTag.textContent = it.kind;
    title.textContent = it.subject;
    const isVid = it.kind === 'Video';
    condImgWrap.hidden = !it.img;
    if (it.img) { condImg.src = it.img; condImgCap.textContent = it.kind === 'Mesh + text' ? 'Input mesh' : 'Input image'; }
    prompt.hidden = isVid; prompt.textContent = '“' + it.cap + '”';
    condRow.hidden = isVid;
    condRow.classList.toggle('is-textonly', !it.img);
    vidCap.textContent = isVid ? 'Input video' : 'Generated video (drives the mesh)';
    side.classList.toggle('is-video', isVid);
  }

  async function select(it) {
    if (current === it.name) return;
    current = it.name; const my = ++token;
    thumbs.forEach((t) => t.setAttribute('aria-current', t._it.name === it.name ? 'true' : 'false'));
    fillSide(it);
    video.src = it.base + 'video.mp4?v=' + VER;
    video.pause();
    status.textContent = 'Loading 3D…'; status.hidden = false; view.classList.add('is-loading');
    try {
      const clip = await loadClip(it.name);
      if (my !== token) return;
      if (!viewer) {
        viewer = new Viewer(canvasHost);
        viewer.clock = () => (scrubbing ? +range.value : frameOf());
        viewer.onInteract = () => hint.classList.add('is-gone');
        viewer.visible = inView;
      }
      viewer.setClip(clip); setGray(segGray.getAttribute('aria-pressed') === 'true');
      range.max = clip.meta.frames - 1;
      status.hidden = true; view.classList.remove('is-loading');
      video.currentTime = 0;
      setPlaying(playing);
    } catch (e) {
      if (my !== token) return;
      status.textContent = 'Could not load the 3D result (' + (window.DecompressionStream ? e.message : 'browser too old') + ').';
    }
  }

  const upd = () => {
    if (viewer && viewer.clip && !scrubbing) { const f = frameOf(); range.value = f; clockTxt.textContent = (f + 1) + ' / ' + frames(); }
    else if (viewer && viewer.clip) clockTxt.textContent = (+range.value + 1) + ' / ' + frames();
    requestAnimationFrame(upd);
  };
  requestAnimationFrame(upd);

  let inView = false, started = false;
  new IntersectionObserver((es) => es.forEach((e) => {
    inView = e.isIntersecting;
    if (inView && !started) { started = true; select(items.find((x) => x.name === Q.get('app')) || items[0]); }
    if (viewer) viewer.visible = inView;
    if (inView && playing) video.play().catch(() => {}); else video.pause();
  }), { rootMargin: '200px 0px' }).observe(stage);
  setPlaying(true); setGray(Q.get('gray') === '1');
}

build();
