#!/bin/zsh
# Fetches the CC0 Poly Haven environment maps (equirectangular HDR, 2k) and table textures used by the
# tier 2 environment-map lighting (docs/HANDOFF_tier2_hdri_envmap.md). They are fetched, not committed.
# Train and held-out rooms are disjoint, for domain randomization with a held-out evaluation set.
set -e
cd "$(dirname "$0")/.."
TRAIN_HDRI=(poly_haven_studio lebombo brown_photostudio_02 artist_workshop small_empty_room_3 hotel_room
            lythwood_room comfy_cafe reading_room pine_attic)
HELDOUT_HDRI=(photo_studio_loft_hall empty_play_room machine_shop_02)
TRAIN_TEX=(wood_table_001 plywood oak_veneer_01 laminate_floor_02)
HELDOUT_TEX=(wood_table_worn)
fetch() {   # fetch <url> <dest>
  [ -s "$2" ] || curl -fsSL -o "$2" "$1"
}
for split in train heldout; do
  mkdir -p assets/polyhaven/hdri/$split assets/polyhaven/tex/$split
  if [ $split = train ]; then H=($TRAIN_HDRI); T=($TRAIN_TEX); else H=($HELDOUT_HDRI); T=($HELDOUT_TEX); fi
  for id in $H; do
    fetch "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/${id}_2k.hdr" assets/polyhaven/hdri/$split/$id.hdr
  done
  for id in $T; do
    url=$(curl -fsSL "https://api.polyhaven.com/files/$id" | python3 -c "import json,sys;print(json.load(sys.stdin)['Diffuse']['1k']['png']['url'])")
    fetch "$url" assets/polyhaven/tex/$split/$id.png
  done
done
ls -R assets/polyhaven | head -40
