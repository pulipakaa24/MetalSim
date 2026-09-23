#!/bin/zsh
# Fetches the Isaac Lab assets used by the parity tests from NVIDIA's public asset store.
# They are NVIDIA's files under NVIDIA's asset terms and are not redistributed in this repository.
set -e
cd "$(dirname "$0")/.."
BASE=${ISAAC_ASSET_BASE:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab}
mkdir -p assets/isaac/G1
[ -f assets/isaac/G1/g1_minimal.usd ] || curl -fL -o assets/isaac/G1/g1_minimal.usd "$BASE/Robots/Unitree/G1/g1_minimal.usd"
echo "g1_minimal.usd md5: $(md5 -q assets/isaac/G1/g1_minimal.usd)  (expected 09bcaf3667df0bde5686f32d74019a97)"
