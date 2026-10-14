#!/bin/zsh
# Fetches the CC0 Poly Haven environment maps (equirectangular HDR) and textures used by the tier 2 environment-map
# lighting (docs/HANDOFF_tier2_hdri_envmap.md) and the ANYmal-on-sand particle demo (scripts/particle_demo/). They are
# fetched, not committed.
# Usage: scripts/fetch_polyhaven_assets.sh [split...]   (default: train heldout)
#   train, heldout: 2k HDRIs and 1k table textures; train and held-out rooms are disjoint, for domain randomization
#                   with a held-out evaluation set
#   demo:           the particle demo's goegap 8k HDRI (~100 MB) and coast_sand_01 2k diffuse texture
# Env: POLYHAVEN (default <repo>/assets/polyhaven), the directory they go to (render_anymal.py reads the same variable).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PH=${POLYHAVEN:-$REPO/assets/polyhaven}
TRAIN_HDRI=(poly_haven_studio lebombo brown_photostudio_02 artist_workshop small_empty_room_3 hotel_room
            lythwood_room comfy_cafe reading_room pine_attic)
HELDOUT_HDRI=(photo_studio_loft_hall empty_play_room machine_shop_02)
TRAIN_TEX=(wood_table_001 plywood oak_veneer_01 laminate_floor_02)
HELDOUT_TEX=(wood_table_worn)
fetch() {   # fetch <url> <dest>
  [ -s "$2" ] || { curl -fsSL -o "$2.part" "$1" && mv "$2.part" "$2"; }
}
api() {   # api <asset id> <python expression over the files JSON `j`>: a download URL from Poly Haven's API
  curl -fsSL "https://api.polyhaven.com/files/$1" | python3 -c "import json,sys; j=json.load(sys.stdin); print($2)"
}
SPLITS=("$@"); [ ${#SPLITS} -eq 0 ] && SPLITS=(train heldout)
for split in $SPLITS; do
  mkdir -p "$PH/hdri/$split" "$PH/tex/$split"
  case $split in
    train)   H=($TRAIN_HDRI); T=($TRAIN_TEX) ;;
    heldout) H=($HELDOUT_HDRI); T=($HELDOUT_TEX) ;;
    demo)
      [ -s "$PH/hdri/demo/goegap_8k.hdr" ] || fetch "$(api goegap "j['hdri']['8k']['hdr']['url']")" "$PH/hdri/demo/goegap_8k.hdr"
      [ -s "$PH/tex/demo/coast_sand_01_diff_2k.png" ] || \
        fetch "$(api coast_sand_01 "j['Diffuse']['2k']['png']['url']")" "$PH/tex/demo/coast_sand_01_diff_2k.png"
      continue ;;
    *) echo "unknown split $split (train | heldout | demo)"; exit 2 ;;
  esac
  for id in $H; do
    fetch "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/${id}_2k.hdr" "$PH/hdri/$split/$id.hdr"
  done
  for id in $T; do
    [ -s "$PH/tex/$split/$id.png" ] || fetch "$(api $id "j['Diffuse']['1k']['png']['url']")" "$PH/tex/$split/$id.png"
  done
done
ls -R "$PH" | head -40
