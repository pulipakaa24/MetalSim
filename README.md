# orchard

Isaac Sim / Isaac Lab parity robotics simulation, native on Apple Silicon.

Design principle (from the plan): match Isaac Sim and Isaac Lab in realism, sensor
fidelity, feature coverage, and per-step efficiency. The only performance gap allowed
is the one caused by the compute, memory bandwidth, and ray-tracing hardware of the
Apple chip in use. Every speed claim is labelled measured, reported, or estimated.

Layout: `orchard/interop` (Metal <-> Warp <-> PyTorch MPS ordering and zero-copy),
`orchard/physics` (MuJoCo Warp on Metal), `orchard/render` (native Metal renderer,
tiers 0-2), `orchard/scene` (USD scene layer, MJCF/URDF importers), `orchard/sensors`,
`orchard/replicator`, `orchard/learn`, `orchard/bench`, `orchard/tools`.
Phase logs and measurements live in `docs/`.
