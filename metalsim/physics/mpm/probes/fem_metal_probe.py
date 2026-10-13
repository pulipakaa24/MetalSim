import warp as wp, warp.fem as fem
wp.init()
print(wp.__file__, wp.config.version, [d.alias for d in wp.get_devices()])
dev = "metal:0"
with wp.ScopedDevice(dev):
    geo = fem.Grid3D(res=wp.vec3i(4,4,4))
    space = fem.make_polynomial_space(geo, degree=1)
    pos = wp.array([[0.3,0.4,0.5],[0.31,0.41,0.52]], dtype=wp.vec3)
    pic = fem.PicQuadrature(fem.Cells(geo), pos)
    test = fem.make_test(space)
    @fem.integrand
    def one(s: fem.Sample, u: fem.Field):
        return u(s)
    for acc in (None, wp.float32):
        try:
            kw = {} if acc is None else {"accumulate_dtype": acc}
            r = fem.integrate(one, quadrature=pic, fields={"u": test}, **kw)
            print("accumulate", acc, "OK sum", float(r.numpy().sum()))
        except Exception as e:
            print("accumulate", acc, "FAILED:", type(e).__name__, str(e)[:300])
