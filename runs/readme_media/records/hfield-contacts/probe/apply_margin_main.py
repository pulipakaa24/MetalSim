"""Apply MuJoCo C main's heightfield margin convention (8626d3c, 2026-10-06) to the fork worktree's ccd_hfield_kernel."""
p = "mujoco_warp/_src/collision_convex.py"; s = open(p).read()
def rep(old, new, n=1):
  global s
  assert s.count(old) == n, (s.count(old), old[:100]); s = s.replace(old, new)
rep("""HFIELD_LOCAL_ORIGIN = True
""", """HFIELD_LOCAL_ORIGIN = True
# MetalSim: heightfield margin convention. True (default): MuJoCo C main's (8626d3c "Fix margin and gap double-counting
# in heightfield collisions", 2026-10-06; test MjcConvexTest.HFieldMarginAndGap): prism and geom each inflated by half of
# margin + gap, contact while the separation is below margin + gap, dist = the separation, as every other CCD pair here.
# False: MuJoCo C 3.14's: prism top raised by margin + gap and geom inflated by half of it, dist = separation -
# 1.5 (margin + gap), contact while the separation is below margin + gap (upstream MuJoCo Warp raises by margin only).
# Identical at margin = gap = 0.
HFIELD_TRUE_DISTANCE = True
""")
# broadphase filter: margin + gap of the two geoms (C main expands the bounds by the margin argument = margin + gap)
rep("""  geom_margin: wp.array2d[float],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  hfield_size: wp.array[wp.vec4],
  # Data in:""", """  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  hfield_size: wp.array[wp.vec4],
  # Data in:""")
rep("""  # TODO(team): margin?
  margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]
""", """  # TODO(team): margin?
  margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]
  # MetalSim: plus the gap, as MuJoCo C (mjc_ConvexHField receives margin + gap)
  margin += geom_gap[worldid % geom_gap.shape[0], g1] + geom_gap[worldid % geom_gap.shape[0], g2]
""")
rep("""      geom_margin,
      mesh_vertadr,
      mesh_vertnum,
      mesh_graphadr,
      mesh_vert,
      mesh_graph,
      hfield_size,
      geom_xpos_in,""", """      geom_margin,
      geom_gap,
      mesh_vertadr,
      mesh_vertnum,
      mesh_graphadr,
      mesh_vert,
      mesh_graph,
      hfield_size,
      geom_xpos_in,""")
rep("""    hfmargin = margin + gap
    geom2.margin = hfmargin
""", """    hfmargin = margin + gap
    geom2.margin = hfmargin
    # prism top raise (MuJoCo C 3.14) or prism inflation (C main: support offset by half the margin, as geom2)
    zraise = float(0.0)
    if wp.static(HFIELD_TRUE_DISTANCE):
      geom1.margin = hfmargin
    else:
      zraise = hfmargin
""")
rep("""    zmin = zmin - org[2]
""", """    zmin = zmin - org[2]
    if wp.static(HFIELD_TRUE_DISTANCE):
      zmin = zmin - hfmargin  # C main expands the geom's bounds by the margin before the prism height test
""")
rep("""        z = hfield_data[adr + (r + dr[init_i]) * ncol + cmin] * size[2] + hfmargin - org[2]
""", """        z = hfield_data[adr + (r + dr[init_i]) * ncol + cmin] * size[2] + zraise - org[2]
""")
rep("""          z = hfield_data[adr + (r + dr[i]) * ncol + c] * size[2] + hfmargin - org[2]
""", """          z = hfield_data[adr + (r + dr[i]) * ncol + c] * size[2] + zraise - org[2]
""")
# plane path: C main: contact while the vertex is within hfmargin of the plane, dist = its distance; C 3.14: raised plane,
# vertex inflated by half the margin
rep("""            best_depth = float(0.0)
            best_p = wp.vec3(0.0, 0.0, 0.0)
            infl = 0.5 * hfmargin
""", """            best_p = wp.vec3(0.0, 0.0, 0.0)
            infl = float(0.0)
            best_depth = float(0.0)
            if wp.static(HFIELD_TRUE_DISTANCE):
              best_depth = hfmargin
            else:
              infl = 0.5 * hfmargin
""")
rep("""            if best_depth >= 0.0:
              continue
            dist = best_depth""", """            if best_depth >= wp.where(wp.static(HFIELD_TRUE_DISTANCE), hfmargin, 0.0):
              continue
            dist = best_depth""")
# GJK/EPA path: keep prisms within the margin (inflated dist < 0), report the separation
rep("""            if ncontact == 0 or dist >= 0.0:
              continue
""", """            if ncontact == 0 or dist >= 0.0:
              continue
            if wp.static(HFIELD_TRUE_DISTANCE):
              dist += hfmargin  # inflated shapes -> separation of the surfaces (both inflated by hfmargin / 2)
""")
open(p, "w").write(s)
print("applied")
