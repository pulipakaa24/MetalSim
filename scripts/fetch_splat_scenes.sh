#!/bin/zsh
# Fetches public 3D Gaussian splat scenes for the splat renderer's real-data tests and benchmark (metalsim/render/SPLAT.md
# §7b). They are fetched, never committed; each real-data test skips with this command when its scene is absent.
# Usage: scripts/fetch_splat_scenes.sh [set...]   (default: interiorgs dxgl)
#   interiorgs    the reference scene: InteriorGS 0204_840158 (showroom, 972,596 splats, SuperSplat compressed .ply,
#                 60 MB) + structure.json (room polygons, used to place cameras); pinned dataset revision and SHA-256.
#                 Licence: InteriorGS Terms of Use (non-commercial research / education, NO redistribution, cite).
#                 Gated: accept the terms once at https://huggingface.co/datasets/spatialverse/InteriorGS with your own
#                 account, then HF_TOKEN or a Hugging Face CLI login (its token file is read; never printed).
#                 INTERIORGS_LOCAL=<dir with 0204_840158/> copies your own existing download instead.
#   dxgl          CC0 1.0. One object splat (nerfstudio splatfacto, standard 3DGS .ply, SH 3) from dxgl/multiview-datasets
#                 (renders of CC0 Poly Haven models): modern_arm_chair.ply. Open download.
#   nurec_lounge  optional, CC-BY-4.0: NVIDIA PhysicalAI-Robotics-NuRec "zh_lounge" (indoor lounge, 3DGUT .nurec,
#                 118 MB). Gated behind a contact-information click-through at
#                 https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec; same token handling.
# Env: METALSIM_SPLAT_DATA (default <repo>/assets/splats), where they go; the tests read the same variable.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DST=${METALSIM_SPLAT_DATA:-$REPO/assets/splats}
# pinned InteriorGS dataset revision and checksum
IGS_REV=334dfeea4e0241033b4e5de97c01bc7c9c080530
IGS_SCENE=0204_840158
IGS_SHA256=c0c748b66a39f0216ecfe7efed7777266ccd7e11013e181bf578c0631f5fdfac

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
fetch() {   # fetch <url> <dest> [header file]
  [ -s "$2" ] && return 0
  if [ -n "$3" ]; then curl -fsSL -H @"$3" -o "$2.part" "$1"; else curl -fsSL -o "$2.part" "$1"; fi
  mv "$2.part" "$2"
}
HDR=""
hf_header() {   # a temporary auth header file for gated Hugging Face downloads (token never echoed)
  [ -n "$HDR" ] && return 0
  local tok=${HF_TOKEN:-} tokfile=${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}
  [ -z "$tok" ] && [ -s "$tokfile" ] && tok=$(<"$tokfile")
  if [ -z "$tok" ]; then
    echo "$1 is gated on Hugging Face: accept its conditions there, then set HF_TOKEN or log in with the Hugging Face CLI." >&2
    return 3
  fi
  HDR=$(mktemp); chmod 600 "$HDR"; trap 'rm -f "$HDR"' EXIT
  print -r -- "Authorization: Bearer $tok" > "$HDR"
}

SETS=("$@"); [ ${#SETS} -eq 0 ] && SETS=(interiorgs dxgl)
for set_ in $SETS; do
  case $set_ in
    dxgl)
      d="$DST/dxgl"; mkdir -p "$d"
      fetch "https://dx.gl/splat/modern_arm_chair.ply" "$d/modern_arm_chair.ply"
      cat > "$d/LICENSE_NOTE.txt" <<'EOF'
dxgl/multiview-datasets (https://huggingface.co/datasets/dxgl/multiview-datasets), CC0-1.0: splats trained on renders
of CC0 Poly Haven models. Fetched by scripts/fetch_splat_scenes.sh.
EOF
      echo "dxgl modern_arm_chair.ply sha256 $(sha "$d/modern_arm_chair.ply")"
      ;;
    nurec_lounge)
      d="$DST/nurec/zh_lounge"; mkdir -p "$d"
      hf_header nvidia/PhysicalAI-Robotics-NuRec || exit 3
      base="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec/resolve/main/zh_lounge/usd"
      for f in export_last.nurec gauss.usda zh_lounge.usda; do fetch "$base/$f" "$d/$f" "$HDR"; done
      cat > "$d/LICENSE_NOTE.txt" <<'EOF'
NVIDIA PhysicalAI-Robotics-NuRec (https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec), zh_lounge, CC-BY-4.0.
Attribution: NVIDIA Corporation. Fetched by scripts/fetch_splat_scenes.sh.
EOF
      echo "nurec zh_lounge export_last.nurec sha256 $(sha "$d/export_last.nurec")"
      ;;
    interiorgs)
      d="$DST/interiorgs/$IGS_SCENE"; mkdir -p "$d"
      if [ -n "$INTERIORGS_LOCAL" ]; then
        for f in 3dgs_compressed.ply structure.json; do [ -s "$d/$f" ] || cp "$INTERIORGS_LOCAL/$IGS_SCENE/$f" "$d/$f"; done
      else
        hf_header spatialverse/InteriorGS || exit 3
        base="https://huggingface.co/datasets/spatialverse/InteriorGS/resolve/$IGS_REV/$IGS_SCENE"
        fetch "$base/3dgs_compressed.ply" "$d/3dgs_compressed.ply" "$HDR"
        fetch "$base/structure.json" "$d/structure.json" "$HDR"
      fi
      got=$(sha "$d/3dgs_compressed.ply")
      [ "$got" = "$IGS_SHA256" ] || { echo "InteriorGS $IGS_SCENE: sha256 $got != pinned $IGS_SHA256" >&2; exit 4; }
      cat > "$DST/interiorgs/LICENSE_NOTE.txt" <<'EOF'
InteriorGS (SpatialVerse / Manycore), https://huggingface.co/datasets/spatialverse/InteriorGS
InteriorGS Terms of Use: non-commercial research and education only; no redistribution of the data in whole or in
part; cite the dataset. Fetched by scripts/fetch_splat_scenes.sh with the user's own token. Never commit these files.
EOF
      ;;
    *) echo "unknown set $set_ (dxgl | nurec_lounge | interiorgs)"; exit 2 ;;
  esac
done
[ -n "$HDR" ] && rm -f "$HDR"
ls -lR "$DST" | head -30
