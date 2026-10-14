"""Apply the ordering-lanes change to a copy of the Warp fork's warp/native/metal.mm and metal.h (research_notes/
gpu_latency/LANES.md).  python3 lanes_patch.py NATIVE_DIR"""
import sys

D = sys.argv[1]


def edit(path, pairs):
    p = f"{D}/{path}"
    s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, "w").write(s)


edit("metal.mm", [
    ("""    id<MTLEvent> event;
    uint64_t event_value = 0;
    id<MTLComputeCommandEncoder> encoder;  // open encoder on command_buffer, if any""",
     """    id<MTLEvent> event;
    uint64_t event_value = 0;
    // Ordering lanes (wp_metal_select_lane): each lane is its own event chain on the same queue, so work on
    // different lanes may run concurrently on the GPU, while work within a lane keeps the order above. `event` /
    // `event_value` always belong to the current lane; the others are parked here. Lane 0 is the device's chain.
    struct Lane {
        id<MTLEvent> event;
        uint64_t value = 0;
    };
    std::vector<Lane> lanes;
    int lane = 0;
    // Set once a second lane was used: frees are then only released at a full synchronize, because command buffers
    // on different lanes no longer complete in commit order (see flush and retire_completed).
    bool multi_lane = false;
    id<MTLComputeCommandEncoder> encoder;  // open encoder on command_buffer, if any"""),
    ("""        dev.in_flight.push_back(dev.command_buffer);
        if (!dev.deferred_frees.empty()) {""",
     """        dev.in_flight.push_back(dev.command_buffer);
        if (!dev.deferred_frees.empty() && !dev.multi_lane) {"""),
    ("""// Diagnostics: n_sync (host waits for the GPU), n_flush (command buffers committed), n_host_ops (host""",
     """// Ordering lanes. Commits the open command buffer of the current lane, then makes `lane` current: later launches
// (and graph replays) are ordered after earlier work of that lane only. Cross-lane order comes from
// wp_metal_event_handle / wp_metal_event_value on one lane and wp_metal_wait_event on the other. Not allowed during a
// capture. Returns 0 on success.
int wp_metal_select_lane(int ordinal, int lane)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || lane < 0 || lane > 15) {
            wp::set_error_string("Invalid Metal device or lane");
            return -1;
        }
        if (dev->capture) {
            wp::set_error_string("Selecting a lane during graph capture is not supported on Metal");
            return -1;
        }
        if (lane == dev->lane)
            return 0;
        if (!flush(*dev))
            return -1;
        if (dev->lanes.empty())
            dev->lanes.push_back({ dev->event, dev->event_value });
        while (int(dev->lanes.size()) <= lane)
            dev->lanes.push_back({ [dev->device newEvent], 0 });
        dev->lanes[dev->lane] = { dev->event, dev->event_value };
        dev->event = dev->lanes[lane].event;
        dev->event_value = dev->lanes[lane].value;
        dev->lane = lane;
        if (lane != 0)
            dev->multi_lane = true;
        return 0;
    }
}

int wp_metal_current_lane(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? dev->lane : -1;
}

// Diagnostics: n_sync (host waits for the GPU), n_flush (command buffers committed), n_host_ops (host"""),
])
edit("metal.h", [
    ("""// Diagnostics: out[0..3] = host waits for the GPU, command buffers committed, host ops run during""",
     """// Ordering lanes: commits the current lane's open work and makes `lane` (0-15) current. Work on different lanes is
// not ordered (it may run concurrently); order them with wp_metal_event_handle/value + wp_metal_wait_event.
WP_API int wp_metal_select_lane(int ordinal, int lane);
WP_API int wp_metal_current_lane(int ordinal);
// Diagnostics: out[0..3] = host waits for the GPU, command buffers committed, host ops run during"""),
])
print("lanes patch applied to", D)
