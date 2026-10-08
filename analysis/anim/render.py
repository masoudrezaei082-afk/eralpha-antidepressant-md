#!/usr/bin/env python3
"""Render an annotated animation of one ligand–ERα–SRC2 complex with Blender Cycles (bpy as a module) and
Molecular Nodes, from the inputs written by mdana.anim (complex.pdb, anim.xtc, frames.json, meta.json).

    python render.py INPUT_DIR OUT_DIR [--samples 64] [--device GPU] [--stills 0,200,500]

Picture (left, W3 x H): receptor cartoon (helix 12 coral, SRC2 gold), pocket residues as sticks, the ligand as
ball-and-stick with carbons in its figure colour, and the ligand–protein H-bonds (amber dashes) and salt bridges
(magenta dashes) of each frame, with residue labels coloured while they interact. The camera starts on the whole
complex, flies into the pocket along the least obstructed direction and then follows the ligand's smoothed centre.
Panel (right): time, the ligand RMSD trace, and a per-residue interaction barcode, each with a moving cursor.

Frames are written as PNG and skipped when present, so an interrupted render resumes; ffmpeg encodes the MP4.
"""
import argparse
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

import numpy as np

W, H, PANEL = 1920, 1080, 600
W3 = W - PANEL
FPS = 30
INTRO, OUTRO = 120, 45            # frames: overview + fly-in, hold at the end
SCALE = 0.01                       # Blender units per Å (Molecular Nodes default)
DIST = 40.0                        # camera distance in the pocket view (Å)
SWING = 22.0                       # camera swings ±SWING° around the up axis over the clip, for depth
T_EXCLUDED = 20.0                  # first 20 ns of production are excluded from the statistics

COL = {"bg": "#F2F4F7", "receptor": "#9DB0CB", "h12": "#E07A5F", "src2": "#E9B949", "res_c": "#97A3B6",
       "anchor_c": "#3FB1A3", "hbond": "#F2A900", "salt": "#D63AF9", "text": "#1F2933", "muted": "#6B7785",
       "N": "#3D6BD6", "O": "#E0453A", "S": "#E2BE28", "F": "#5DBB46", "Cl": "#2E9E44", "Br": "#9C4A1E",
       "H": "#F4F4F4", "P": "#F08A24"}
LABEL_HB, LABEL_SB = "#A86A00", "#9C1FA6"   # darker versions of the dash colours for legible text


def rgb(hexc):
    h = hexc.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def lin(hexc):
    c = rgb(hexc)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def ligand_carbon(hexc):
    """The figure colour, lifted when it is too dark to read as a lit surface (estradiol is black in the figures)."""
    c = rgb(hexc)
    return "#3C3C3C" if c.max() < 0.2 else hexc


def smooth(x, sigma):
    k = np.arange(-3 * sigma, 3 * sigma + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    pad = np.pad(x, ((3 * sigma, 3 * sigma), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], w, mode="valid") for j in range(x.shape[1])], 1)


def unit(v):
    return v / np.linalg.norm(v)


def rotate(v, axis, deg):
    """Rodrigues rotation of v about a unit axis."""
    a = math.radians(deg)
    return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * np.dot(axis, v) * (1 - math.cos(a))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Data:
    def __init__(self, inp):
        import MDAnalysis as mda
        self.meta = json.loads((inp / "meta.json").read_text())
        self.frames = json.loads((inp / "frames.json").read_text())
        self.u = mda.Universe(str(inp / "complex.pdb"), str(inp / "anim.xtc"))
        u, m = self.u, self.meta
        self.n = u.trajectory.n_frames
        assert self.n == len(self.frames) == m["n_frames"], (self.n, len(self.frames))
        self.xyz = np.stack([u.atoms.positions.copy() for _ in u.trajectory]).astype(np.float32)
        self.lig = u.atoms[m["ligand_ix"]]
        self.time = np.array([f["time_ns"] for f in self.frames])
        self.rmsd = np.array([f["lig_rmsd"] for f in self.frames])
        r0, r1 = m["receptor_resid"]
        p0, p1 = m["peptide_resid"]
        self.rec = u.select_atoms(f"resid {r0}:{r1}")
        self.pep = u.select_atoms(f"resid {p0}:{p1}")
        self.h12 = u.select_atoms("resid {}:{} and name CA".format(*m["h12_resid"]))
        shown = [r["resid"] for r in m["shown_residues"]]
        self.shown = shown
        self.pocket_ca = u.select_atoms("name CA and resid " + " ".join(map(str, shown)))
        self.side = {}
        for r in shown:
            ag = u.select_atoms(f"resid {r} and not backbone and not element H")
            self.side[r] = ag if ag.n_atoms else u.select_atoms(f"resid {r} and name CA")
        self.resid_of = u.atoms.resids
        # per frame and tracked residue: 0 none, 1 H-bond, 2 salt bridge, 3 both
        self.tracked = m["tracked_residues"][:10]
        row = {r["resid"]: i for i, r in enumerate(self.tracked)}
        self.code = np.zeros((self.n, len(self.tracked)), dtype=int)
        self.res_state = []
        for k, f in enumerate(self.frames):
            st = {}
            for d, h, a in f["hbonds"]:
                prot_atom = a if a not in set(m["ligand_ix"]) else d
                r = int(self.resid_of[prot_atom])
                st[r] = st.get(r, 0) | 1
            for li, ai in f["salt"]:
                r = int(self.resid_of[ai])
                st[r] = st.get(r, 0) | 2
            self.res_state.append(st)
            for r, v in st.items():
                if r in row:
                    self.code[k, row[r]] = v
        self.lig_cog = np.stack([self.xyz[k][self.lig.ix].mean(0) for k in range(self.n)])

    def sec_struct(self, step=5):
        """Per-atom Molecular Nodes sec_struct codes (1 helix, 2 strand, 3 loop, 0 not protein) from DSSP averaged
        over every step-th frame. Molecular Nodes' own DSSP needs equal numbers of N, CA, C and O atoms, which the
        Amber termini (OXT, or OC1/OC2) break, so only residues with exactly one of each are passed to DSSP."""
        from MDAnalysis.analysis.dssp import DSSP, translate
        prot = self.rec + self.pep
        keep = [r for r in prot.residues
                if all((r.atoms.names == nm).sum() == 1 for nm in ("N", "CA", "C", "O"))]
        bb = sum((r.atoms.select_atoms("name N CA C O") for r in keep[1:]), keep[0].atoms.select_atoms("name N CA C O"))
        run = DSSP(bb).run(step=step)
        chars = translate(run.results.dssp_ndarray.mean(axis=0))
        code = {"H": 1, "E": 2, "-": 3}
        per_res = np.zeros(len(self.u.residues), dtype=int)
        per_res[prot.residues.resindices] = 3
        per_res[[r.resindex for r in keep]] = [code[c] for c in chars]
        print("DSSP: {} helix, {} strand, {} loop residues".format(*(int((per_res == v).sum()) for v in (1, 2, 3))),
              flush=True)
        return per_res[self.u.atoms.resindices].astype(np.int32)


class Camera:
    """Overview -> pocket fly-in, then a slow swing around the up axis while following the smoothed ligand."""

    def __init__(self, d: Data):
        x0 = d.xyz[0]
        self.pocket = x0[d.pocket_ca.ix].mean(0)
        self.centre = x0[d.u.select_atoms("name CA").ix].mean(0)
        self.view = unit(np.array(d.meta["view_dir"]))
        up = x0[d.pep.ix].mean(0) - self.pocket          # SRC2 towards the top of the picture
        up = up - np.dot(up, self.view) * self.view
        if np.linalg.norm(up) < 5.0:
            up = np.cross(self.view, [1.0, 0, 0]) if abs(self.view[0]) < 0.9 else np.cross(self.view, [0, 1.0, 0])
        self.up = unit(up)
        self.lig_s = smooth(d.lig_cog, 12)
        self.n = d.n

    def at(self, k):
        """(camera position, look-at target, up) in Å for output frame k (intro frames are k < 0)."""
        if k < 0:
            s = ease((k + INTRO - 25) / (INTRO - 30))
            tgt_end, dist_end = self._pocket_target(0)
            tgt = (1 - s) * self.centre + s * tgt_end
            dist = (1 - s) * 165.0 + s * dist_end
            ang = (1 - s) * -55.0
        else:
            kk = min(k, self.n - 1)
            tgt, dist = self._pocket_target(kk)
            ang = SWING * math.sin(2 * math.pi * kk / (self.n - 1))
        direction = rotate(self.view, self.up, ang)
        return tgt + direction * dist, tgt, self.up

    def _pocket_target(self, k):
        drift = self.lig_s[k] - self.pocket
        return self.pocket + 0.6 * drift, DIST + 0.6 * np.linalg.norm(drift)


def set_attr(obj_entity, name, data):
    try:
        obj_entity.store_named_attribute(data=data, name=name, atype="FLOAT_COLOR")
    except TypeError:
        obj_entity.store_named_attribute(data, name)


def emission_material(name, hexc, strength):
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    c = (*lin(hexc), 1.0)
    bsdf.inputs["Base Color"].default_value = c
    bsdf.inputs["Emission Color"].default_value = c
    bsdf.inputs["Emission Strength"].default_value = strength
    bsdf.inputs["Roughness"].default_value = 0.4
    return mat


def fade_material(base):
    """Copy of a Molecular Nodes material that turns transparent closer to the camera than a cut depth (set per
    frame by set_cut). This opens the protein in front of the pocket without the black interiors a clipping plane
    leaves on the cut tubes."""
    mat = base.copy()
    mat.name = base.name + " cutaway"
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
    link = out.inputs["Surface"].links[0]
    src = link.from_socket
    nt.links.remove(link)
    cam, rng = nt.nodes.new("ShaderNodeCameraData"), nt.nodes.new("ShaderNodeMapRange")
    clear, mix = nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader")
    rng.clamp = True
    rng.name = "cut"
    nt.links.new(cam.outputs["View Z Depth"], rng.inputs["Value"])
    nt.links.new(rng.outputs["Result"], mix.inputs[0])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(src, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def dash_mesh(segs, radius, dash, gap, trim, nside=10):
    """Vertices and faces of dashed cylinders along each (p, q) segment, in Å."""
    verts, faces = [], []
    th = np.linspace(0, 2 * np.pi, nside, endpoint=False)
    for p, q in segs:
        v = q - p
        length = np.linalg.norm(v)
        if length < 2 * trim + 0.2:
            continue
        dvec = v / length
        a = np.cross(dvec, [1.0, 0, 0]) if abs(dvec[0]) < 0.9 else np.cross(dvec, [0, 1.0, 0])
        a = unit(a)
        b = np.cross(dvec, a)
        ring = np.outer(np.cos(th), a) + np.outer(np.sin(th), b)
        t, end = trim, length - trim
        n_dash = max(1, int(round((end - t + gap) / (dash + gap))))
        step = (end - t) / n_dash
        for i in range(n_dash):
            s0, s1 = t + i * step, t + i * step + step * dash / (dash + gap)
            base = len(verts)
            for s in (s0, s1):
                verts.extend(p + dvec * s + radius * ring)
            for j in range(nside):
                jj = (j + 1) % nside
                faces.append((base + j, base + jj, base + nside + jj, base + nside + j))
            faces.append(tuple(base + j for j in range(nside))[::-1])
            faces.append(tuple(base + nside + j for j in range(nside)))
    return np.array(verts).reshape(-1, 3), faces


class Scene:
    def __init__(self, d: Data, a):
        import bpy
        import molecularnodes as mn
        self.bpy, self.d = bpy, d
        m = d.meta
        self.canvas = mn.Canvas(engine="CYCLES", resolution=(W3, H), transparent=False)
        # the engine is configured after the template has loaded, which would otherwise reset samples and device
        self.canvas.engine = mn.scene.Cycles(samples=a.samples, device=a.device, denoise=True)
        if a.device.upper() == "GPU":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            print("Cycles device:", prefs.compute_device_type, [dv.name for dv in prefs.devices if dv.use], flush=True)
        self.canvas.view_transform = "Standard"       # so the background matches the overlay colours exactly
        sc = bpy.context.scene
        world = sc.world
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        out, path = nt.nodes.new("ShaderNodeOutputWorld"), nt.nodes.new("ShaderNodeLightPath")
        bg_cam, bg_light = nt.nodes.new("ShaderNodeBackground"), nt.nodes.new("ShaderNodeBackground")
        mix = nt.nodes.new("ShaderNodeMixShader")
        bg_cam.inputs["Color"].default_value = (*lin(COL["bg"]), 1.0)
        bg_light.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_light.inputs["Strength"].default_value = a.ambient
        nt.links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
        nt.links.new(bg_light.outputs[0], mix.inputs[1])
        nt.links.new(bg_cam.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        sc.render.use_persistent_data = True
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.max_bounces = 4
        sc.render.film_transparent = False

        u = d.u
        traj = mn.Trajectory(u, name=m["key"])
        traj.update_with_scene = False
        self.traj = traj
        # secondary structure (cartoon helices and strands), fixed for the clip so it does not flicker
        ss = d.sec_struct()
        traj.calculations["sec_struct"] = lambda _u: ss

        n = u.atoms.n_atoms
        resid, el = u.atoms.resids, u.atoms.elements
        cart = np.tile(lin(COL["receptor"]), (n, 1))
        lo, hi = m["h12_resid"]
        cart[(resid >= lo) & (resid <= hi)] = lin(COL["h12"])
        p0, p1 = m["peptide_resid"]
        cart[(resid >= p0) & (resid <= p1)] = lin(COL["src2"])
        anchors = {r["resid"] for r in m["shown_residues"] if r["anchor"]}
        atom = np.tile(lin(COL["res_c"]), (n, 1))
        atom[np.isin(resid, list(anchors))] = lin(COL["anchor_c"])
        atom[d.lig.ix] = lin(ligand_carbon(m["color"]))
        for e, hexc in COL.items():
            if len(e) <= 2 and e[0].isupper():
                atom[el == e] = lin(hexc)
        ones = np.ones((n, 1))
        set_attr(traj, "anim_cartoon", np.hstack([cart, ones]))
        set_attr(traj, "anim_atom", np.hstack([atom, ones]))

        polar_h = [a.ix for a in d.lig if a.element == "H" and any(b.element in ("N", "O") for b in a.bonded_atoms)]
        lig_sel = d.lig.select_atoms("not element H") + u.atoms[polar_h]
        res_sel = u.select_atoms("resid {} and (not backbone or name CA) and not element H".format(
            " ".join(map(str, d.shown))))
        prot = d.rec + d.pep
        from molecularnodes.nodes.material import append_material
        base = bpy.data.materials.get("MN Default") or append_material("MN Default")
        self.fade = fade_material(base)
        traj.add_style(mn.StyleCartoon(), color="anim_cartoon", selection=prot, material=self.fade, name="cartoon")
        traj.add_style(mn.StyleSticks(radius=0.2), color="anim_atom", selection=res_sel, name="pocket")
        traj.add_style(mn.StyleBallAndStick(sphere_radius=0.27, bond_radius=0.15), color="anim_atom",
                       selection=lig_sel, name="ligand")

        # interaction dashes: one mesh per kind, rebuilt every frame
        self.dash = {}
        for kind, strength in (("hbond", 2.0), ("salt", 2.0)):
            me = bpy.data.meshes.new(kind)
            ob = bpy.data.objects.new(kind, me)
            sc.collection.objects.link(ob)
            ob.data.materials.append(emission_material(kind, COL[kind], strength))
            self.dash[kind] = ob

        cam = sc.camera
        cam.data.lens = 50
        cam.data.sensor_fit = "HORIZONTAL"
        cam.data.clip_end = 50.0
        if a.fstop > 0:
            cam.data.dof.use_dof = True
            cam.data.dof.aperture_fstop = a.fstop
        self.cam = cam
        # sun lights (brightness independent of distance, so the overview and the pocket view are lit alike) ride
        # with the camera: a key from up and to the left and a dim rim from behind, both aimed at the target
        self.light_target = bpy.data.objects.new("light_target", None)
        sc.collection.objects.link(self.light_target)
        for name, off, energy in (("key", (-0.25, 0.30, 0.10), a.light), ("rim", (0.20, 0.10, -0.80), a.light * 0.4)):
            ld = bpy.data.lights.new(name, "SUN")
            ld.energy, ld.angle = energy, math.radians(12)
            ob = bpy.data.objects.new(name, ld)
            sc.collection.objects.link(ob)
            ob.parent = cam
            ob.location = off
            con = ob.constraints.new("TRACK_TO")
            con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
            con.target = self.light_target
        self.sc = sc

    def set_cut(self, depth):
        """Cartoon closer to the camera than depth (Å) is not drawn; it fades in over the next 1.5 Å."""
        rng = self.fade.node_tree.nodes["cut"]
        rng.inputs["From Min"].default_value = depth * SCALE
        rng.inputs["From Max"].default_value = (depth + 1.5) * SCALE

    def place_camera(self, pos, tgt, up, clip_start):
        from mathutils import Matrix, Vector
        z = unit(pos - tgt)                    # camera looks down -Z
        x = unit(np.cross(up, z))
        y = np.cross(z, x)
        m = Matrix(((x[0], y[0], z[0], pos[0] * SCALE), (x[1], y[1], z[1], pos[1] * SCALE),
                    (x[2], y[2], z[2], pos[2] * SCALE), (0, 0, 0, 1)))
        self.cam.matrix_world = m
        self.cam.data.clip_start = clip_start * SCALE
        self.cam.data.dof.focus_distance = float(np.linalg.norm(pos - tgt)) * SCALE
        self.light_target.location = Vector(tgt * SCALE)

    def set_frame(self, k):
        d = self.d
        self.traj.frame = k
        self.traj.set_frame(k)
        xyz = d.xyz[k]
        segs = {"hbond": [], "salt": []}
        for dn, h, ac in d.frames[k]["hbonds"]:
            segs["hbond"].append((xyz[dn], xyz[ac]))          # donor to acceptor, both heavy atoms
        for li, ai in d.frames[k]["salt"]:
            segs["salt"].append((xyz[li], xyz[ai]))
        for kind, ob in self.dash.items():
            radius = 0.14 if kind == "hbond" else 0.16
            v, f = dash_mesh(segs[kind], radius, dash=0.30, gap=0.20, trim=0.45)
            me = ob.data
            me.clear_geometry()
            if len(v):
                me.from_pydata((v * SCALE).tolist(), [], f)
            me.update()

    def project(self, pts):
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector
        out = []
        for p in pts:
            c = world_to_camera_view(self.sc, self.cam, Vector(p * SCALE))
            out.append((c.x * W3, (1 - c.y) * H, c.z / SCALE))
        return out

    def render(self, path):
        self.sc.render.filepath = str(path)
        self.sc.render.image_settings.file_format = "PNG"
        self.bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------- panel (matplotlib, static) + per-frame overlay

@functools.lru_cache(maxsize=None)
def font(bold=False, mono=False, size=20):
    from PIL import ImageFont
    import matplotlib
    cands = []
    if mono:
        cands += ["/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
                  str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono-Bold.ttf")]
    name = "Bold" if bold else "Regular"
    cands += [f"/usr/share/fonts/truetype/liberation/LiberationSans-{name}.ttf",
              str(Path(matplotlib.get_data_path()) / f"fonts/ttf/DejaVuSans{'-Bold' if bold else ''}.ttf")]
    for c in cands:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def build_panel(d: Data, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lib = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
    if lib.exists():
        from matplotlib import font_manager
        for f in lib.parent.glob("LiberationSans-*.ttf"):
            font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "Liberation Sans"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "text.color": COL["text"],
                         "axes.labelcolor": COL["text"], "xtick.color": COL["muted"], "ytick.color": COL["muted"]})
    m = d.meta
    fig = plt.figure(figsize=(PANEL / 100, H / 100), dpi=100, facecolor="white")
    fig.add_artist(Line2D([0.004, 0.004], [0, 1], color="#D5DAE1", lw=2))
    fig.add_artist(Line2D([0.05, 0.05], [0.912, 0.968], color=ligand_carbon(m["color"]), lw=7,
                          solid_capstyle="butt"))
    fig.text(0.08, 0.952, m["label"], fontsize=27, weight="bold", va="center")
    fig.text(0.08, 0.922, "bound to ERα ligand-binding domain + SRC2 peptide", fontsize=11.5, va="center",
             color=COL["muted"])
    fig.text(0.05, 0.884, f"Replica {m['replica']} of 3 (median ligand RMSD) · 100 ns · 300 K", fontsize=10.5,
             va="center", color=COL["muted"])

    # ligand RMSD trace
    ax = fig.add_axes([0.13, 0.565, 0.80, 0.165])
    ax.axvspan(0, T_EXCLUDED, color="#E9ECF1", lw=0)
    ax.axhline(2.0, color=COL["muted"], lw=0.8, ls=(0, (4, 3)))
    ax.plot(d.time, d.rmsd, color=ligand_carbon(m["color"]), lw=1.4)
    ymax = max(4.0, math.ceil(float(np.nanmax(d.rmsd)) + 0.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Time (ns)", fontsize=10.5, labelpad=2)
    ax.set_ylabel("RMSD (Å)", fontsize=10.5)
    ax.text(T_EXCLUDED / 2, ymax * 0.93, "excluded\nfrom stats", fontsize=8, ha="center", va="top",
            color=COL["muted"])
    ax.text(99, 2.0, "2 Å", fontsize=8.5, ha="right", va="bottom", color=COL["muted"])
    fig.text(0.05, 0.762, "Ligand RMSD to its starting pose", fontsize=13, weight="bold")

    # interaction barcode
    nrow = len(d.tracked)
    top, rowh = 0.455, min(0.042, 0.28 / nrow)
    bx = fig.add_axes([0.27, top - nrow * rowh, 0.58, nrow * rowh])
    cmap = ListedColormap(["#EEF1F5", COL["hbond"], COL["salt"], COL["salt"]])
    t = d.time
    bx.imshow(d.code.T, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
              extent=(t[0] - (t[1] - t[0]) / 2, t[-1] + (t[1] - t[0]) / 2, nrow - 0.5, -0.5))
    bx.set_yticks(range(nrow))
    bx.set_yticklabels([r["label"].replace("SRC2-", "SRC2 ") for r in d.tracked], fontsize=10.5)
    for lab, r in zip(bx.get_yticklabels(), d.tracked):
        if r["anchor"]:
            lab.set_fontweight("bold")
            lab.set_color(COL["text"])
    bx.set_xlim(0, 100)
    bx.set_xlabel("Time (ns)", fontsize=10.5, labelpad=2)
    bx.tick_params(axis="y", length=0, pad=4)
    for sp in bx.spines.values():
        sp.set_visible(False)
    eq = d.time >= T_EXCLUDED
    for i in range(nrow):
        frac = float((d.code[eq, i] > 0).mean())
        bx.text(1.02, i, f"{frac * 100:.0f}%", transform=bx.get_yaxis_transform(), fontsize=9.5,
                va="center", color=COL["muted"])
    fig.text(0.05, top + 0.040, "Ligand H-bonds and salt bridges", fontsize=13, weight="bold")
    fig.text(0.05, top + 0.017, "bold: Glu353 · Arg394 · His524 (estradiol anchors);  % = 20–100 ns",
             fontsize=9, color=COL["muted"])
    fig.text(0.95, top + 0.017, "now", fontsize=9, color=COL["muted"], ha="right")

    # legend
    lx = fig.add_axes([0.04, max(0.075, top - nrow * rowh - 0.185), 0.92, 0.105])
    lx.axis("off")
    hand = [Patch(color=rgb(COL["receptor"]), label="ERα LBD"),
            Patch(color=rgb(COL["h12"]), label="Helix 12 (AF-2)"),
            Patch(color=rgb(COL["src2"]), label="SRC2 coactivator"),
            Patch(color=rgb(ligand_carbon(m["color"])), label=f"{m['label']} carbons"),
            Line2D([], [], color=COL["hbond"], lw=3, ls=(0, (1.2, 0.8)), label="H-bond"),
            Line2D([], [], color=COL["salt"], lw=3, ls=(0, (1.2, 0.8)), label="Salt bridge"),
            Patch(color=rgb(COL["anchor_c"]), label="Anchor residue carbons"),
            Patch(color=rgb(COL["res_c"]), label="Other pocket residues")]
    lx.legend(handles=hand, ncol=2, loc="upper left", frameon=False, fontsize=10, handlelength=1.6,
              columnspacing=1.2, labelspacing=0.55, borderaxespad=0)
    c = m["criteria"]
    foot = (f"H-bond: {c['hbond']}. Salt bridge: cationic N to carboxylate O ≤ 4.0 Å. Drawn when present "
            f"{c['present']}; each picture frame averages {2 * m['half_window_frames'] + 1} frames saved every "
            f"{m['raw_dt_ps']:g} ps. Pocket-fitted trajectory. One simulation of three; simulations are "
            "hypothesis-generating and do not measure binding affinity.")
    fig.text(0.05, 0.012, textwrap.fill(foot, 98), fontsize=8, color=COL["muted"], va="bottom", linespacing=1.35)

    fig.canvas.draw()
    geo = {"rmsd_x0": ax.transData.transform((0, 0))[0], "rmsd_x1": ax.transData.transform((100, 0))[0]}
    y0, y1 = ax.transData.transform((0, 0))[1], ax.transData.transform((0, ymax))[1]
    geo.update(rmsd_ytop=H - y1, rmsd_ybot=H - y0, rmsd_ymax=ymax)
    geo["bar_x0"], geo["bar_x1"] = bx.transData.transform((0, 0))[0], bx.transData.transform((100, 0))[0]
    geo["rows"] = [H - bx.transData.transform((0, i))[1] for i in range(nrow)]
    geo["bar_ytop"], geo["bar_ybot"] = H - bx.transData.transform((0, -0.5))[1], \
        H - bx.transData.transform((0, nrow - 0.5))[1]
    fig.savefig(out_png, dpi=100, facecolor="white")
    plt.close(fig)
    return geo


def compose(d, scene, render_png, panel, geo, k_data, intro_phase, out_png):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), "white")
    img.paste(Image.open(render_png).convert("RGB"), (0, 0))
    img.paste(panel, (W3, 0))
    dr = ImageDraw.Draw(img)
    m = d.meta
    t = float(d.time[k_data])

    # residue labels at the side-chain centre once the camera is in the pocket, coloured while the residue
    # interacts; placed greedily (interacting, then anchors, then the rest) and nudged or dropped to avoid overlaps
    xyz = d.xyz[k_data]
    st = d.res_state[k_data]
    f_lab, f_lab_b = font(size=21), font(bold=True, size=22)
    # the ligand's screen box is kept free; each label is pushed outward from the ligand, then nudged sideways
    lig_px = np.array([q[:2] for q in scene.project(xyz[d.lig.ix])])
    lc = lig_px.mean(0)
    placed = [] if intro_phase else [(*(lig_px.min(0) - 14), *(lig_px.max(0) + 14))]
    items = []
    for r in (m["shown_residues"] if not intro_phase else []):
        ag = d.side[r["resid"]]
        x, y, z = scene.project([xyz[ag.ix].mean(0)])[0]
        if z <= 0 or not (40 < x < W3 - 40 and 30 < y < H - 30):
            continue
        s_ = st.get(r["resid"], 0)
        items.append((0 if s_ else 1 if r["anchor"] else 2, x, y, r, s_))
    for prio, x, y, r, s_ in sorted(items, key=lambda it: it[0]):
        fnt = f_lab_b if (s_ or r["anchor"]) else f_lab
        fill = LABEL_SB if s_ & 2 else LABEL_HB if s_ & 1 else COL["text"]
        out = np.array([x, y]) - lc
        out = out / (np.linalg.norm(out) + 1e-6)
        side = np.array([-out[1], out[0]])
        # residues behind the ligand need the long pushes to clear its box
        for push, lat in ((22, 0), (40, 0), (22, 26), (22, -26), (62, 0), (40, 40), (40, -40), (90, 0),
                          (120, 0), (120, 45), (120, -45), (150, 0), (180, 0), (150, 70), (150, -70), (220, 0)):
            px, py = np.array([x, y]) + out * push + side * lat
            bb = dr.textbbox((px, py), r["label"], font=fnt, anchor="mm", stroke_width=4)
            if bb[0] < 10 or bb[2] > W3 - 10 or bb[1] < 10 or bb[3] > H - 50:
                continue
            if all(bb[2] < q[0] or bb[0] > q[2] or bb[3] < q[1] or bb[1] > q[3] for q in placed):
                placed.append(bb)
                dr.text((px, py), r["label"], font=fnt, fill=fill, anchor="mm", stroke_width=4, stroke_fill="white")
                break
    for name, ag, hexc in (("Helix 12", d.h12, COL["h12"]), ("SRC2", d.pep.select_atoms("name CA"), "#B07F0A")):
        x, y, z = scene.project([xyz[ag.ix].mean(0)])[0]
        if z > 0 and 60 < x < W3 - 60 and 40 < y < H - 40:
            dr.text((x, y), name, font=font(bold=True, size=26), fill=hexc, anchor="mm", stroke_width=5,
                    stroke_fill="white")

    # caption in the picture
    cap = ("ERα ligand-binding domain · coactivator-bound (SRC2) conformation" if intro_phase
           else "pocket view · camera follows the ligand")
    dr.text((28, H - 28), cap, font=font(size=20), fill=COL["muted"], anchor="ls", stroke_width=3,
            stroke_fill=COL["bg"])

    # time and progress
    ox = W3
    dr.text((ox + 30, 214), f"t = {t:.1f} ns", font=font(mono=True, size=38), fill=COL["text"], anchor="ls")
    tag = ("starting pose" if intro_phase else
           f"0–{T_EXCLUDED:g} ns: excluded from statistics" if t < T_EXCLUDED else "")
    if tag:
        dr.text((ox + PANEL - 30, 212), tag, font=font(size=15), fill=COL["muted"], anchor="rs")
    x0, x1, yb = ox + 30, ox + PANEL - 30, 230
    dr.rounded_rectangle((x0, yb, x1, yb + 8), radius=4, fill="#E3E7ED")
    dr.rounded_rectangle((x0, yb, x0 + max(8, (x1 - x0) * t / 100), yb + 8), radius=4,
                         fill=ligand_carbon(m["color"]))

    # cursors
    cx = ox + geo["rmsd_x0"] + (geo["rmsd_x1"] - geo["rmsd_x0"]) * t / 100
    dr.line((cx, geo["rmsd_ytop"], cx, geo["rmsd_ybot"]), fill="#1F2933", width=2)
    val = float(d.rmsd[k_data])
    cy = geo["rmsd_ybot"] - (geo["rmsd_ybot"] - geo["rmsd_ytop"]) * min(val, geo["rmsd_ymax"]) / geo["rmsd_ymax"]
    dr.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=ligand_carbon(m["color"]), outline="white", width=2)
    dr.text((ox + PANEL - 30, 302), f"{val:.1f} Å", font=font(bold=True, size=20), fill=COL["text"], anchor="rs")
    bx = ox + geo["bar_x0"] + (geo["bar_x1"] - geo["bar_x0"]) * t / 100
    dr.line((bx, geo["bar_ytop"] - 3, bx, geo["bar_ybot"] + 3), fill="#1F2933", width=2)
    for i, ry in enumerate(geo["rows"]):
        code = int(d.code[k_data, i])
        x = ox + PANEL - 42
        if code:
            dr.ellipse((x - 8, ry - 8, x + 8, ry + 8), fill=COL["salt"] if code & 2 else COL["hbond"])
        else:
            dr.ellipse((x - 7, ry - 7, x + 7, ry + 7), outline="#B9C1CC", width=2)
    img.save(out_png, optimize=False, compress_level=3)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("inp", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--device", default="GPU")
    p.add_argument("--fstop", type=float, default=8.0, help="0 disables depth of field")
    p.add_argument("--light", type=float, default=0.9, help="key sun strength")
    p.add_argument("--ambient", type=float, default=0.45, help="world (ambient) light strength")
    p.add_argument("--stills", default="", help="comma-separated output frames to render as stills only")
    p.add_argument("--max-intro-s", type=float, default=0,
                   help="exit 3 (slow node) if the first 12 overview frames average more than this; 0 disables")
    a = p.parse_args()

    d = Data(a.inp)
    key = d.meta["key"]
    frames_dir = a.out / f"{key}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp = a.out / f"{key}_render.png"
    sc = Scene(d, a)
    camera = Camera(d)
    from PIL import Image
    geo = build_panel(d, a.out / f"{key}_panel.png")
    panel = Image.open(a.out / f"{key}_panel.png").convert("RGB").resize((PANEL, H))

    total = INTRO + d.n + OUTRO
    todo = [int(x) for x in a.stills.split(",")] if a.stills else range(total)
    t0, n_done = time.time(), 0
    for j in todo:
        out_png = frames_dir / f"{j:05d}.png"
        if out_png.exists():
            continue
        k_cam = j - INTRO
        k_data = min(max(k_cam, 0), d.n - 1)
        pos, tgt, up = camera.at(min(k_cam, d.n - 1))
        # cut-away: nothing in the overview, then the cartoon in front of the pocket is removed during the last
        # 40 intro frames, down to 11 Å in front of the look-at point
        dist = float(np.linalg.norm(pos - tgt))
        s = ease((k_cam + 40) / 40)
        sc.set_cut(s * (dist - 11.0))
        sc.set_frame(k_data)
        sc.place_camera(pos, tgt, up, 1.0)
        sc.render(tmp)
        compose(d, sc, tmp, panel, geo, k_data, k_cam < 0, out_png)
        n_done += 1
        # the overview frames cost the same on every system, so a node far slower than its GPU class should be is
        # found here (a throttled or shared card ran 9x slower) and given up before it bills for hours
        if a.max_intro_s and n_done == 12 and k_cam < 0 and (time.time() - t0) / n_done > a.max_intro_s:
            print(f"{key}: slow node, {(time.time() - t0) / n_done:.1f} s/frame on the overview "
                  f"(limit {a.max_intro_s:g})", flush=True)
            sys.exit(3)
        if n_done % 25 == 0 or a.stills:
            el = time.time() - t0
            print(f"{key}: frame {j}/{total}, {el / n_done:.2f} s/frame", flush=True)

    if a.stills:
        return
    mp4 = a.out / f"{key}.mp4"
    ff = shutil.which("ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(mp4)], check=True)
    shutil.copyfile(frames_dir / f"{INTRO + d.n // 2:05d}.png", a.out / f"{key}_poster.png")
    print(f"{key}: wrote {mp4} ({total} frames, {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    # bpy 4.5 deadlocks in interpreter finalisation (after its leak-detector messages), so the process would never
    # exit and the node would keep billing: leave through os._exit once the output is flushed
    rc = 0
    try:
        main()
    except BaseException as e:  # noqa: BLE001 - report and exit non-zero, never fall through to finalisation
        traceback.print_exc()
        rc = e.code if isinstance(e, SystemExit) and isinstance(e.code, int) else 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
