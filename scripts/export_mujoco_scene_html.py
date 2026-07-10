#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export a static RL-100 MuJoCo scene to a rotatable Three.js HTML viewer.")
    parser.add_argument("--suite", choices=["metaworld", "adroit"], default="metaworld")
    parser.add_argument("--task", default="door-unlock")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def make_sim(args):
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.pop("PYOPENGL_PLATFORM", None)

    if args.suite == "metaworld":
        import metaworld

        task_name = args.task
        if "-v2" not in task_name:
            task_name = f"{task_name}-v2-goal-observable"
        env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        env._freeze_rand_vec = False
        env.reset()
        env.sim.forward()
        return env, env.sim

    from rl_100.env import AdroitEnv

    env = AdroitEnv(env_name=args.task, use_point_cloud=False)
    env.reset()
    sim = env.get_mujoco_sim()
    sim.forward()
    return env, sim


def matrix_to_list(matrix):
    return np.asarray(matrix, dtype=np.float64).reshape(-1).tolist()


def geom_name(model, idx):
    try:
        return model.geom_id2name(idx) or f"geom_{idx}"
    except Exception:
        return f"geom_{idx}"


def mesh_name(model, idx):
    try:
        return model.mesh_id2name(idx) or f"mesh_{idx}"
    except Exception:
        return f"mesh_{idx}"


def material_for_geom(model, idx):
    rgba = np.asarray(model.geom_rgba[idx], dtype=np.float64)
    if rgba[3] <= 0:
        rgba = np.array([0.65, 0.65, 0.65, 1.0])
    return rgba.tolist()


def export_scene(sim):
    model = sim.model
    data = sim.data
    geoms = []

    for i in range(model.ngeom):
        geom_type = int(model.geom_type[i])
        if geom_type == 0:
            continue

        pos = np.asarray(data.geom_xpos[i], dtype=np.float64)
        mat = np.asarray(data.geom_xmat[i], dtype=np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[i], dtype=np.float64)
        rgba = material_for_geom(model, i)
        base = {
            "name": geom_name(model, i),
            "type": geom_type,
            "pos": pos.tolist(),
            "mat": matrix_to_list(mat),
            "size": size.tolist(),
            "rgba": rgba,
        }

        data_id = int(model.geom_dataid[i])
        if geom_type == 7 and data_id >= 0:
            v_start = int(model.mesh_vertadr[data_id])
            v_end = v_start + int(model.mesh_vertnum[data_id])
            f_start = int(model.mesh_faceadr[data_id])
            f_end = f_start + int(model.mesh_facenum[data_id])
            vertices = np.asarray(model.mesh_vert[v_start:v_end], dtype=np.float64)
            faces = np.asarray(model.mesh_face[f_start:f_end], dtype=np.int64)
            if faces.min(initial=0) >= v_start:
                faces = faces - v_start
            base.update({
                "kind": "mesh",
                "mesh_name": mesh_name(model, data_id),
                "vertices": vertices.reshape(-1).tolist(),
                "faces": faces.reshape(-1).tolist(),
            })
        else:
            base["kind"] = "primitive"
        geoms.append(base)

    return {
        "geoms": geoms,
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
    }


def html_template(scene_json, title):
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; height: 100%; overflow: hidden; background: #f4f5f7; font-family: system-ui, sans-serif; }}
    #hud {{ position: absolute; left: 12px; top: 10px; padding: 8px 10px; background: rgba(255,255,255,.88); border: 1px solid #ddd; border-radius: 6px; color: #222; font-size: 13px; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
<div id="hud">{title}<br>Left drag: rotate · Right drag: pan · Wheel: zoom</div>
<script>
const sceneData = {scene_json};
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f5f7);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(1.4, -2.0, 1.2);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({{antialias: true}});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.4);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 0.8));
const light = new THREE.DirectionalLight(0xffffff, 0.75);
light.position.set(2, -3, 5);
scene.add(light);

const axes = new THREE.AxesHelper(0.35);
scene.add(axes);

function mat4FromGeom(g) {{
  const m = new THREE.Matrix4();
  const e = g.mat;
  m.set(
    e[0], e[1], e[2], g.pos[0],
    e[3], e[4], e[5], g.pos[1],
    e[6], e[7], e[8], g.pos[2],
    0, 0, 0, 1
  );
  return m;
}}

function material(g) {{
  return new THREE.MeshStandardMaterial({{
    color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]),
    opacity: g.rgba[3],
    transparent: g.rgba[3] < 0.99,
    roughness: 0.55,
    metalness: 0.05,
    side: THREE.DoubleSide
  }});
}}

function primitiveGeometry(g) {{
  const s = g.size;
  if (g.type === 2) return new THREE.SphereGeometry(s[0], 32, 16);
  if (g.type === 3) return new THREE.CapsuleGeometry(s[0], Math.max(0.001, 2 * s[1]), 12, 24);
  if (g.type === 5) return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32);
  if (g.type === 6) return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  return new THREE.SphereGeometry(Math.max(s[0], 0.01), 16, 8);
}}

const bbox = new THREE.Box3();
for (const g of sceneData.geoms) {{
  let geom;
  if (g.kind === "mesh") {{
    geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(g.vertices), 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(g.faces), 1));
    geom.computeVertexNormals();
  }} else {{
    geom = primitiveGeometry(g);
  }}
  const mesh = new THREE.Mesh(geom, material(g));
  mesh.name = g.name;
  mesh.applyMatrix4(mat4FromGeom(g));
  scene.add(mesh);
  bbox.expandByObject(mesh);
}}

if (!bbox.isEmpty()) {{
  const center = new THREE.Vector3();
  bbox.getCenter(center);
  controls.target.copy(center);
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 0.5);
  camera.position.set(center.x + radius, center.y - radius * 1.5, center.z + radius * 0.8);
  controls.update();
}}

window.addEventListener("resize", () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    output = args.output
    if output is None:
        safe_task = args.task.replace("/", "_")
        output = f"outputs/{args.suite}_{safe_task}_static_scene.html"

    env, sim = make_sim(args)
    try:
        scene = export_scene(sim)
    finally:
        if hasattr(env, "close"):
            env.close()

    scene_json = json.dumps(scene, separators=(",", ":"))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_template(scene_json, f"{args.suite} {args.task} static scene"), encoding="utf-8")
    print(path.resolve())
    print(f"exported geoms={len(scene['geoms'])} meshes={scene['nmesh']}")


if __name__ == "__main__":
    main()
