"""Geodetic placement and the live simulation server.

The failure mode here is an aircraft rendered over the wrong ground: a globe
viewer will happily draw a perfectly smooth track a hundred kilometres from
where the simulation actually put it, and nothing looks broken. So the transform
is checked against published landmark elevations, not just for self-consistency.
"""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.data.geodetic import GeoRef
from naigos.env.flight_env import NaigosEnv

pytest.importorskip("pyproj")


@pytest.fixture(scope="module")
def theatre():
    from naigos.env.theatre_bridge import env_from_theatre

    try:
        return env_from_theatre(aoi="tehran_basin", n_blue=4, n_threat=10)
    except FileNotFoundError as e:
        pytest.skip(f"tehran_basin snapshot unavailable: {e}")


def test_enu_to_wgs84_round_trips_exactly(theatre):
    cfg, _, notes = theatre
    g = GeoRef(**notes["georef"])
    xs = np.linspace(0, cfg.terrain.extent_x, 11)
    ys = np.linspace(0, cfg.terrain.extent_y, 11)
    lon, lat = g.to_wgs84(xs, ys)
    x2, y2 = g.from_wgs84(lon, lat)
    assert np.abs(x2 - xs).max() < 1e-3
    assert np.abs(y2 - ys).max() < 1e-3


def test_theatre_lands_on_the_right_part_of_the_planet(theatre):
    _, _, notes = theatre
    b = notes["geo_bounds"]
    # Tehran is ~35.7 N, 51.4 E. A sign flip or a wrong UTM zone puts this in
    # the wrong hemisphere and the check fails loudly.
    assert 50.5 < b["west"] < 52.5 and 50.5 < b["east"] < 52.5
    assert 35.0 < b["south"] < 36.5 and 35.0 < b["north"] < 36.5
    assert b["east"] > b["west"] and b["north"] > b["south"]


@pytest.mark.parametrize(
    "name,lon,lat,published_m,tol_m",
    [
        ("Azadi / central Tehran", 51.3379, 35.6997, 1250, 120),
        ("Mehrabad apron", 51.3134, 35.6892, 1208, 120),
    ],
)
def test_dem_elevation_matches_published_landmarks(theatre, name, lon, lat, published_m, tol_m):
    """Independent ground truth for the whole chain: bucket -> reprojection ->
    ENU resample -> georef. A datum or zone error shows up as hundreds of metres."""
    from naigos.env.terrain import sample_height

    cfg, hmap, notes = theatre
    g = GeoRef(**notes["georef"])
    x, y = g.from_wgs84(lon, lat)
    assert 0 <= x <= cfg.terrain.extent_x and 0 <= y <= cfg.terrain.extent_y, f"{name} outside AOI"
    h = float(sample_height(hmap, cfg.terrain, jnp.float32(x), jnp.float32(y)))
    assert abs(h - published_m) < tol_m, f"{name}: DEM {h:.0f} m vs published {published_m} m"


def test_a_wrong_utm_zone_would_be_caught(theatre):
    """Proves the landmark check has teeth."""
    _, _, notes = theatre
    bad = GeoRef(**{**notes["georef"], "utm_epsg": notes["georef"]["utm_epsg"] + 1})
    lon_ok, lat_ok = GeoRef(**notes["georef"]).to_wgs84(0.0, 0.0)
    lon_bad, lat_bad = bad.to_wgs84(0.0, 0.0)
    assert abs(float(lon_bad) - float(lon_ok)) > 1.0


# ---------------------------------------------------------------- respawn ---
def _env():
    from naigos.env.config import EnvConfig

    return NaigosEnv(EnvConfig(n_blue=4, n_threat=8, n_threat_active=6, max_steps=60))


def test_respawn_clears_only_the_retasked_aircraft_tracks():
    """A fresh aircraft must not inherit the track history of the one that just
    died at the far end of the map -- it would be shot down seconds after
    spawning, for reasons invisible in the trace."""
    env = _env()
    st, _ = env.reset(jax.random.PRNGKey(0))
    st = st._replace(lock=jnp.ones_like(st.lock) * 0.9, dwell=jnp.ones_like(st.dwell) * 30.0)
    mask = jnp.array([False, True, True, False])
    st2 = env.respawn(st, jax.random.PRNGKey(1), mask)

    assert float(st2.lock[:, 1].max()) == 0.0 and float(st2.dwell[:, 2].max()) == 0.0
    assert float(st2.lock[:, 0].max()) == pytest.approx(0.9)
    assert bool(jnp.all(st2.alive))
    assert float(st2.air.fuel[1]) == env.cfg.airframe.fuel_init


def test_respawn_leaves_untouched_aircraft_alone():
    env = _env()
    st, _ = env.reset(jax.random.PRNGKey(0))
    st2 = env.step(st, jnp.full((4, 3), 0.4))[0]
    st3 = env.respawn(st2, jax.random.PRNGKey(2), jnp.array([True, False, False, False]))
    assert jnp.allclose(st3.air.pos[1:], st2.air.pos[1:])
    assert not jnp.allclose(st3.air.pos[0], st2.air.pos[0])


def test_respawn_and_reroll_are_jit_able():
    env = _env()
    st, _ = env.reset(jax.random.PRNGKey(0))
    jax.jit(env.respawn)(st, jax.random.PRNGKey(1), jnp.array([True, False, True, False]))
    jax.jit(env.reroll_threats)(st, jax.random.PRNGKey(2))


def test_reroll_moves_threats_clears_tracks_and_keeps_terrain():
    """Without this the whole live session reports one threat layout."""
    env = _env()
    st, _ = env.reset(jax.random.PRNGKey(0))
    st = st._replace(lock=jnp.ones_like(st.lock))
    st2 = env.reroll_threats(st, jax.random.PRNGKey(9))
    assert not jnp.allclose(st.threats.pos, st2.threats.pos)
    assert float(st2.lock.max()) == 0.0
    assert jnp.allclose(st.hmap, st2.hmap)


def test_live_simulation_carries_and_resets_recurrent_policy_memory():
    """The frame-by-frame path must honor the same lifecycle as env.rollout."""
    from naigos.demo.live import Simulation
    from naigos.rl.ppo import PPOConfig, greedy_policy, init_learner

    env = _env()
    key = jax.random.PRNGKey(41)
    _, obs = env.reset(key)
    ppo = PPOConfig(policy_family="recurrent", recurrent_seq_len=4)
    learner = init_learner(key, env.cfg, ppo, obs)
    sim = Simulation(
        env=env,
        policy=greedy_policy(learner.actor.params, env.cfg, family="recurrent"),
        georef=GeoRef(utm_epsg=32611, origin_easting_m=400_000.0,
                      origin_northing_m=4_000_000.0),
        reroll_s=0.0,
    )
    np.testing.assert_array_equal(np.asarray(sim.policy_memory), 0.0)

    out = sim._step(sim.state, sim.obs, sim.policy_memory, key)
    state2, obs2, memory2, *_ = out
    assert memory2.shape == sim.policy_memory.shape
    assert bool(np.any(np.asarray(memory2) != 0.0))

    # A max-step world boundary resets every aircraft before another frame can
    # consume the state. Individual respawns reset only their own row.
    terminal = state2._replace(t=env.cfg.max_steps - 1)
    out = sim._step(terminal, obs2, memory2, key)
    np.testing.assert_array_equal(np.asarray(out[2]), 0.0)
    sim.policy_memory = jnp.ones_like(memory2)
    sim._reset_policy_memory(np.array([True, False, True, False]))
    host_memory = np.asarray(sim.policy_memory)
    np.testing.assert_array_equal(host_memory[[0, 2]], 0.0)
    np.testing.assert_array_equal(host_memory[[1, 3]], 1.0)
    sim._reset_policy_memory()  # threat-world reroll
    np.testing.assert_array_equal(np.asarray(sim.policy_memory), 0.0)


def test_live_run_loop_resets_recurrent_memory_on_world_reroll():
    """`run()` itself must call the reset, not just the helper."""
    from naigos.demo.live import Simulation
    from naigos.rl.ppo import PPOConfig, greedy_policy, init_learner

    env = _env()
    key = jax.random.PRNGKey(43)
    _, obs = env.reset(key)
    learner = init_learner(key, env.cfg, PPOConfig(policy_family="recurrent"), obs)
    sim = Simulation(
        env=env,
        policy=greedy_policy(learner.actor.params, env.cfg, family="recurrent"),
        georef=GeoRef(utm_epsg=32611, origin_easting_m=400_000.0,
                      origin_northing_m=4_000_000.0),
        speed=1e6,
        reroll_s=1e-6,
    )
    calls, frames = [], []
    real = sim._reset_policy_memory
    sim._reset_policy_memory = lambda done_mask=None: (calls.append(done_mask), real(done_mask))
    sim._publish = lambda *a, **k: (frames.append(1), len(frames) >= 2 and sim.stop())
    sim.run()
    assert any(c is None for c in calls), "world reroll did not clear GRU memory"


# ------------------------------------------------------------------ server ---
def test_live_frames_are_wellformed_and_geolocated(theatre):
    """Drive the Simulation directly -- no socket, no thread, no flake."""
    from pathlib import Path

    from naigos.demo.live import Simulation
    from naigos.rl.checkpoint import load as load_checkpoint
    from naigos.rl.ppo import greedy_policy

    ckpt = Path("checkpoints/theatre_1000")
    if not ckpt.exists():
        pytest.skip("no shipped checkpoint")
    cfg, hmap, notes = theatre
    params = load_checkpoint(ckpt)["actor"]
    sim = Simulation(
        env=NaigosEnv(cfg, hmap=hmap),
        policy=greedy_policy(params, cfg),
        georef=GeoRef(**notes["georef"]),
        speed=1e9,          # no wall-clock pacing in a test
        reroll_s=0.0,
    )
    for _ in range(3):
        sim.key, k = jax.random.split(sim.key)
        s2, o2, terms, info, feas, _ = sim._step(sim.state, sim.obs, k)
        sim.state, sim.obs = s2, o2
        sim._publish(np.asarray(feas), np.asarray(terms.exposure))

    f = sim.latest()
    json.dumps(f)  # must be serialisable as-is
    b = notes["geo_bounds"]
    for a in f["aircraft"]:
        assert b["west"] - 0.2 <= a["lon"] <= b["east"] + 0.2
        assert b["south"] - 0.2 <= a["lat"] <= b["north"] + 0.2
        assert 0.0 <= a["heading"] < 360.0
        assert math.isfinite(a["alt"]) and math.isfinite(a["agl"])
    assert set(f["counters"]) >= {"sorties_launched", "objectives_reached", "shot_down"}


def test_scene_payload_is_serialisable_and_complete(theatre):
    from pathlib import Path

    from naigos.demo.live import Simulation
    from naigos.rl.checkpoint import load as load_checkpoint
    from naigos.rl.ppo import greedy_policy

    ckpt = Path("checkpoints/theatre_1000")
    if not ckpt.exists():
        pytest.skip("no shipped checkpoint")
    cfg, hmap, notes = theatre
    sim = Simulation(
        env=NaigosEnv(cfg, hmap=hmap),
        policy=greedy_policy(load_checkpoint(ckpt)["actor"], cfg),
        georef=GeoRef(**notes["georef"]),
        reroll_s=0.0,
    )
    scene = sim.scene(notes)
    json.dumps(scene)
    assert len(scene["threats"]) == cfg.n_threat
    assert scene["theatre"] == "tehran_basin"
    for t in scene["threats"]:
        assert 50.0 < t["lon"] < 53.0 and 34.0 < t["lat"] < 37.0


def test_cesium_page_has_one_pinned_external_script():
    from pathlib import Path

    html = (Path("naigos/demo/assets/cesium.html")).read_text()
    import re

    srcs = re.findall(r'<script[^>]*src="([^"]+)"', html)
    assert srcs == ["https://cesium.com/downloads/cesiumjs/releases/1.145/Build/Cesium/Cesium.js"]
    assert "http://" not in html.replace("http://127.0.0.1", "")
    assert "__ION_TOKEN__" in html  # the substitution point the server fills
