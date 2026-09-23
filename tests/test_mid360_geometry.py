import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sensor"))
import mid360_geometry as geo  # noqa: E402

LO, HI = geo.MID360["elev_min_deg"], geo.MID360["elev_max_deg"]


def _window_deg(n_up, az_deg):
    return [math.degrees(v) for v in geo.elevation_window(n_up, math.radians(az_deg), LO, HI)]


def test_rpy_convention_and_inverted_mount_axis():
    # URDF rpy is Rz(yaw) Ry(pitch) Rx(roll); the main.urdf mount equals Rx(pi) Ry(-delta).
    R = geo.rpy_to_matrix([0.0, 3.101, 3.1415])
    delta = math.pi - 3.101
    assert np.allclose(R, geo.rot_x(math.pi) @ geo.rot_y(-delta), atol=1e-4)
    assert R[2, 2] < -0.99 and R[0, 2] < 0  # sensor z points down, tilted backwards
    q = np.array([math.cos(0.3), 0.0, math.sin(0.3), 0.0])  # 0.6 rad about y
    assert np.allclose(geo.quat_wxyz_to_matrix(q), geo.axis_angle_matrix([0, 1, 0], 0.6))
    pitch, roll = geo.pitch_roll_of(geo.rot_y(0.2) @ geo.rot_x(0.1))
    assert pitch == pytest.approx(0.2) and roll == pytest.approx(0.1)


def test_elevation_window_level_inverted_and_tilted():
    for az in (0, 90, 200):
        lo, hi, glo, _ = _window_deg([0, 0, 1], az)
        assert (lo, hi) == pytest.approx((LO, HI)) and math.isnan(glo)
        lo, hi, _, _ = _window_deg([0, 0, -1], az)
        assert (lo, hi) == pytest.approx((-HI, -LO))
    t = math.radians(5.0)  # sensor axis tilted 5 deg forward: the front band drops by 5 deg
    lo, hi, _, _ = _window_deg([math.sin(t), 0, math.cos(t)], 0)
    assert (lo, hi) == pytest.approx((LO - 5, HI - 5))
    lo, hi, _, _ = _window_deg([math.sin(t), 0, math.cos(t)], 180)
    assert (lo, hi) == pytest.approx((LO + 5, HI + 5))


def test_elevation_window_gap_when_axis_is_horizontal():
    # Axis pointing forward: sensor elevation = 90 - |e|, so |e| >= 38 deg is inside the band.
    lo, hi, glo, ghi = _window_deg([1, 0, 0], 0)
    assert (lo, hi) == pytest.approx((-90, 90))
    assert (glo, ghi) == pytest.approx((-38, 38))


def test_elevation_window_matches_brute_force():
    rng = np.random.default_rng(1)
    E = np.radians(np.linspace(-90, 90, 18001))
    for _ in range(25):
        R = geo.quat_wxyz_to_matrix(rng.normal(size=4))
        for az in np.radians([0.0, 75.0, 180.0, 300.0]):
            inside = geo.in_fov(geo.direction(az, E) @ R, LO, HI)
            lo, hi, _, _ = geo.elevation_window(R[:, 2], az, LO, HI)
            if not inside.any():
                assert np.isnan(lo)
                continue
            assert lo == pytest.approx(E[inside].min(), abs=2e-4)
            assert hi == pytest.approx(E[inside].max(), abs=2e-4)


def test_fov_sampling_is_uniform_in_solid_angle():
    d = geo.sample_fov_directions(20000, LO, HI)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert geo.in_fov(d, LO, HI).all()
    z = d[:, 2]
    mid = (math.sin(math.radians(LO)) + math.sin(math.radians(HI))) / 2
    assert z.mean() == pytest.approx(mid, abs=1e-3)
    az, _ = geo.az_el(d)
    assert np.histogram(az, 8)[0].std() / 2500 < 0.02


def test_ray_sphere_and_capsule():
    o = np.zeros(3)
    d = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]])
    t = geo.ray_sphere(o, d, [2.0, 0, 0], 0.5)
    assert t[0] == pytest.approx(1.5) and np.isinf(t[1:]).all()
    # Capsule along y at x=2: body hit, cap hit and miss.
    a, b = np.array([2.0, -0.5, 0]), np.array([2.0, 0.5, 0])
    dirs = np.array([[1.0, 0, 0], [2.0, 0.7, 0], [0, 0, 1.0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    t = geo.ray_capsule(o, dirs, a, b, 0.2)
    assert t[0] == pytest.approx(1.8)
    hit = dirs[1] * t[1]
    seg = a + np.clip((hit - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1) * (b - a)
    assert np.linalg.norm(hit - seg) == pytest.approx(0.2)
    assert np.isinf(t[2])
    assert (geo.ray_capsule([2.0, 0, 0], dirs, a, b, 0.2) == 0).all()  # origin inside


def test_ray_capsule_matches_marching():
    rng = np.random.default_rng(2)
    dirs = rng.normal(size=(400, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    o, a, b = np.array([0.1, -0.2, 0.3]), np.array([0.5, 0.2, 0.1]), np.array([0.2, 0.6, 0.4])
    r = 0.15
    t = geo.ray_capsule(o, dirs, a, b, r)
    s = np.linspace(0, 2, 4001)
    for d, ti in zip(dirs, t):
        p = o + s[:, None] * d
        h = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        inside = np.linalg.norm(p - (a + h[:, None] * (b - a)), axis=1) <= r
        if inside.any():
            assert ti == pytest.approx(s[np.argmax(inside)], abs=1e-3)
        else:
            assert np.isinf(ti) or ti > 1.99


def test_ray_box_rotated():
    R = geo.rot_z(math.radians(45))
    d = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    t = geo.ray_box(np.zeros(3), d, [2.0, 0, 0], R, [0.5, 0.5, 0.5])
    assert t[0] == pytest.approx(2.0 - 0.5 * math.sqrt(2)) and np.isinf(t[1])
    assert (geo.ray_box([2.0, 0, 0], d, [2.0, 0, 0], R, [0.5] * 3) == 0).all()


def test_floor_and_overhead_geometry():
    assert geo.floor_blind_radius(1.2, math.radians(-45)) == pytest.approx(1.2)
    assert np.isinf(geo.floor_blind_radius(1.2, math.radians(5)))
    lo, hi = math.radians(-54.3), math.radians(4.7)
    assert geo.overhead_last_visible(0.2, lo, hi) == pytest.approx(0.2 / math.tan(hi))
    assert geo.overhead_last_visible(-0.2, lo, hi) == pytest.approx(0.2 / math.tan(-lo))
    assert np.isinf(geo.overhead_last_visible(0.2, lo, math.radians(-3)))
    # Corridor: any point with |y| <= 0.3 m counts, so the edge stays visible closer in.
    d = geo.corridor_last_visible(np.array([0, 0, 1.0]), 1.0, 1.5)
    assert d == pytest.approx(math.sqrt((0.5 / math.tan(math.radians(HI))) ** 2 - 0.09), abs=0.011)


def test_approach_last_seen():
    thr = np.full(500, 1.0)
    last = geo.approach_last_seen(thr, 0.02, speed=0.6, scan_hz=10.0, n_arrivals=200)
    assert (last >= 1.0).all() and (last < 1.0 + 0.06 + 1e-9).all()
    assert np.isinf(geo.approach_last_seen(np.full(10, np.inf), 0.02, n_arrivals=5)).all()
    # Alternating poses: the favourable one wins.
    thr = np.where(np.arange(500) % 50 < 25, 0.5, 3.0)
    assert np.median(geo.approach_last_seen(thr, 0.02, n_arrivals=500)) < 1.0


URDF = """<robot name="toy">
  <link name="base"/><link name="arm"/><link name="tip"/>
  <joint name="j1" type="revolute"><origin xyz="0 0 1" rpy="0 0 0"/>
    <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint>
  <joint name="{mount}" type="fixed"><origin xyz="{x} 0 0" rpy="0 0 0"/>
    <parent link="arm"/><child link="tip"/></joint>
</robot>"""


def test_fk_mount_and_transplant(tmp_path):
    a = tmp_path / "a.urdf"
    a.write_text(URDF.format(mount="mid360_joint", x=0.5))
    model = geo.UrdfModel.load(a)
    poses = model.fk({"j1": np.array([0.0, math.pi / 2])}, np.zeros((2, 3)), np.eye(3)[None])
    assert np.allclose(poses["tip"][1], [[0.5, 0, 1], [0, 0.5, 1]])
    mount = geo.mount_from_urdf(a, "m")
    R, p = mount.pose(*poses["arm"])
    assert np.allclose(p, poses["tip"][1])
    # A second file whose "arm" frame sits 0.1 m higher but puts the sensor at the same point.
    b = tmp_path / "b.urdf"
    b.write_text(URDF.format(mount="mid360_joint", x=0.5).replace('xyz="0 0 1"', 'xyz="0 0 1.1"')
                 .replace('xyz="0.5 0 0"', 'xyz="0.5 0 -0.1"'))
    moved = geo.transplant_mount(b, model, "t")
    assert np.allclose(moved.xyz, [0.5, 0, 0])


def test_stl_reader_and_proxy_fits(tmp_path):
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 1], [0, 1, 1]]], float)
    blob = b"\0" * 80 + struct.pack("<I", 2)
    for t in tri:
        blob += struct.pack("<12fH", 0, 0, 1, *t.reshape(-1), 0)
    f = tmp_path / "t.stl"
    f.write_bytes(blob)
    assert np.allclose(geo.read_stl(f), tri)
    pts = np.random.default_rng(3).normal(size=(500, 3)) * [0.3, 0.05, 0.05]
    a, b, r = geo.fit_capsule(pts)
    cap = geo.Proxy("x", "capsule", a, b, r)
    assert all(cap.contains(p) for p in pts * (1 - 1e-9))
    assert np.linalg.norm(b - a) > 0.5  # elongated cloud keeps a real segment
    c_s, r_s = geo.fit_spheres(pts, 6)
    dist = np.linalg.norm(pts[:, None] - c_s[None], axis=2)
    assert np.all((dist <= r_s[None] + 1e-9).any(1))  # quantile 1 spheres cover every point
    c, R, half = geo.fit_box(pts)
    assert np.all(np.abs((pts - c) @ R) <= half + 1e-9)


def test_clip_kind():
    assert geo.clip_kind("A person walks slowly and carefully across the room.") == "gait"
    assert geo.clip_kind("A person takes small careful steps forward.") == "gait"
    assert geo.clip_kind("A person walks to a door, opens it, and walks through.") == "other"
    assert geo.clip_kind("An old person walks forward, pauses to look around.") == "other"
    assert geo.clip_kind("A person ducks under a low hanging obstacle while walking.") == "crouch"
    assert geo.clip_kind("A person crawls forward on their hands and knees.") == "crawl"


def test_fk_matches_reference_bodies_when_data_present():
    try:
        ws = geo.find_workspace()
        urdf = geo.sonic_asset(geo.URDF_REL, ws)
    except FileNotFoundError:
        pytest.skip("workspace data not on this host")
    ref_path = ws / "m2s-hindsight-dataset-v1-20260911/motions/00102/reference.npz"
    if not ref_path.exists():
        pytest.skip("reference clip not on this host")
    model = geo.UrdfModel.load(urdf)
    ref = geo.load_reference(ref_path)
    poses = model.fk(ref["q"], ref["root_pos"], ref["root_R"])
    z = np.load(ref_path)
    for i, name in enumerate(z["body_names"]):
        assert np.abs(poses[str(name)][1] - z["body_position_m"][:, i]).max() < 1e-5
    inv = geo.mount_from_urdf(urdf, "inverted")
    up = geo.transplant_mount(geo.REPO / geo.G1_29DOF_REL, model, "upright")
    assert np.allclose(inv.xyz, up.xyz, atol=1e-9)  # same point, different orientation
    stand = geo.default_clip(model)
    fr = geo.sensor_frames(model, inv, stand, np.zeros((1, 3)))
    assert fr["height"][0] == pytest.approx(1.217, abs=0.005)
    assert fr["front_lo"][0] == pytest.approx(-54.33, abs=0.01)
    assert fr["front_hi"][0] == pytest.approx(4.67, abs=0.01)
