#!/bin/zsh
# Fetches the Isaac Lab assets used by the parity tests and tasks from NVIDIA's public asset store.
# They are NVIDIA's files under NVIDIA's asset terms and are not redistributed in this repository.
set -e
cd "$(dirname "$0")/.."
BASE=${ISAAC_ASSET_BASE:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab}
mkdir -p assets/isaac/G1
[ -f assets/isaac/G1/g1_minimal.usd ] || curl -fL -o assets/isaac/G1/g1_minimal.usd "$BASE/Robots/Unitree/G1/g1_minimal.usd"
echo "g1_minimal.usd md5: $(md5 -q assets/isaac/G1/g1_minimal.usd)  (expected 09bcaf3667df0bde5686f32d74019a97)"

# Franka-Pour (IsaacContrib-Franka-Pour, metalsim/learn/franka_pour*.py): the Isaac 6.1 asset root Isaac Lab 3.0-EA
# reads (apps/isaaclab.python.headless.kit: persistent.isaac.asset_root.default). The Franka is a layered USD (root +
# payloads); the published 6.2 root serves byte-identical files. SHA-256 of each file as fetched on 2026-10-11.
ROOT61=${ISAAC_ASSET_ROOT_61:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac}
FP=assets/isaac/franka_pour
mkdir -p $FP/franka/payloads/Physics $FP/table
fetch() {   # fetch <url> <dest> <sha256>
  [ -f "$2" ] || curl -fsSL -o "$2" "$1"
  local h=$(shasum -a 256 "$2" | cut -d' ' -f1)
  [ "$h" = "$3" ] || { echo "SHA-256 mismatch for $2: $h (expected $3)"; exit 1; }
}
fetch $ROOT61/IsaacLab/Contrib/MPM/Pour/franka_pour_cups.usda $FP/franka_pour_cups.usda e762038b4e024aacb17fb65e25f3b5d631e95fc93213804f0b034eda0b0b2ac1
fetch $ROOT61/IsaacLab/Contrib/MPM/Pour/reset_dataset.pt $FP/reset_dataset.pt 044a9986669be27bd884f1f7299961d011ed5e3286c007d65773fffe5dd00aa6
fetch $ROOT61/Props/Mounts/SeattleLabTable/table_instanceable.usd $FP/table/table_instanceable.usd 479ed8fd374d8eafdedf3ca3b9d0d5981ffc6d5c5016929347f748be93d5df17
fetch $ROOT61/Props/Mounts/SeattleLabTable/table.usd $FP/table/table.usd 8e6e2284e0eba868341b5afacf663d4f84005febef4f9c178d916e609d44c114
F=$ROOT61/IsaacLab/Robots/FrankaEmika
fetch $F/franka_panda.usda $FP/franka/franka_panda.usda eaf27723ba1934d792c6876679eb75491bdff944ad108d4e59734fa07dc65809
fetch $F/payloads/base.usda $FP/franka/payloads/base.usda 17091b4d8e3136901541d497c0c8fe671a3192f5f4763e709193d477fc9618d0
fetch $F/payloads/base_flat_hulls.usda $FP/franka/payloads/base_flat_hulls.usda 9cfd0feccf60022c9c1fd08c5d2e25035f4196f7ade03c2cecd0fc4bea21f4a2
fetch $F/payloads/base_flat_primitives.usda $FP/franka/payloads/base_flat_primitives.usda 12d94dbad7b60dfa794d9717f9f82f311911f2b4ba0334ee89bb49168d4ca4e7
fetch $F/payloads/geometries.usd $FP/franka/payloads/geometries.usd c64f015a69302f5b277040fc54fc7addedab49c47e67d1b79f374414dd4e991b
fetch $F/payloads/instances_physx_hulls_clean.usda $FP/franka/payloads/instances_physx_hulls_clean.usda 50254bc251e240439ae5bcce491c8d84ef5eae6da21d6ff5115b113b368b0a4d
fetch $F/payloads/instances.usda $FP/franka/payloads/instances.usda d151d980980479a03818fe6be4ba983a99b9a5482caa6de233d11fd5bd247726
fetch $F/payloads/materials.usda $FP/franka/payloads/materials.usda 3b87896947047adc23abafac72ccb113b26a32293645db8e755c51f2abe58b9b
fetch $F/payloads/robot.usda $FP/franka/payloads/robot.usda 060bd8bc6f47b6d57ef918d05629af36763998a275c5102bd4aa4540e0fcf15b
fetch $F/payloads/Physics/mujoco.usda $FP/franka/payloads/Physics/mujoco.usda 82f363dd872bd8ca88847a76b1daf267fdace4095f6d505f3eb16acf49596f8d
fetch $F/payloads/Physics/physics.usda $FP/franka/payloads/Physics/physics.usda 0dc38454f02ea14d9ddd2437995fdc7c4a65634443cacdcc2a04e3de25655e00
fetch $F/payloads/Physics/physx.usda $FP/franka/payloads/Physics/physx.usda cf5d9c127149518114dcdb84fac18ff89bb4700bb9aabe9e803114e85372f070
echo "Franka-Pour assets: $FP (15 files, SHA-256 verified)"
