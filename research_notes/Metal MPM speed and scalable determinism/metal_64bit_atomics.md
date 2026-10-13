# 64-bit compare-and-swap / exchange without 64-bit CAS: Metal-safe protocols for Newton's remaining sites

Researched 2026-10-11. "Measured" means a probe run on this machine: Apple M4 Max (Apple9 family, 40-core GPU), macOS
Darwin 25.6, GPUCompiler 32023, the integrated-2 Warp tree (`warp-i2`, the code that refuses 64-bit CAS on Metal,
same refusal as fork `metalsim` 152808b7), through `scripts/gpu_run.sh` (kind `check`). Probe sources and logs are in
the session scratchpad, not in the repo. Everything else is cited or read from the code (file and branch named).

## Summary

* Apple GPUs have exactly two 64-bit atomics: `atomic_min_explicit` / `atomic_max_explicit` on `device atomic_ulong`.
  They take only `memory_order_relaxed` and **return nothing**. There is no 64-bit load, store, exchange, CAS,
  fetch_min/max or add, and no `atomic_long` (measured below; MSL 4.1 sections 6.16.4.4 and 6.16.4.6 as quoted by
  Apple DTS).
* The workable toolbox is: (1) **pack-and-max/min**, where the value and a tie-breaker share one 64-bit word and the
  hardware min/max is the whole publish; (2) **claim-then-publish**: a 32-bit CAS on one half claims a slot, one
  64-bit min/max publishes the full word; (3) **monotone slots**: if every write to a word is a max, a value that has
  been displaced never returns, so a stale or torn read can only err in the safe direction; (4) **count, then
  finalize in a second launch** where a single kernel cannot decide the winner.
* Waiting on another thread is the hazard. Apple GPUs have no documented forward-progress guarantee between
  threadgroups, and an empirical study found they do not follow even the occupancy-bound model. SIMT compilers can
  also reconverge a loop so that a waiting lane starves the lane it waits for. Every retry loop below is therefore
  **bounded**, and a thread that runs out of passes reports a failure instead of hanging.
* Per-site designs (section 4): contact_reduction_global finalize by `atomic_max` plus read-back, rollback skipped on
  Metal. XPBD restitution manifold table: the HashTable claim/publish with a tag bit, empty = 0 kept, bounded passes.
  Kamino argmax: a packed `(value, ~tid)` 64-bit max per world plus a per-world finalize launch. The
  test_contact_reduction_global "errors" are library calls, not direct CAS: they go away with the library fix (one
  test helper needs to ignore provisional id 0). Prototypes of the first three protocols pass on the M4 Max
  (section 5).

## Q1. What Metal offers (cited, then measured)

### Cited
- MSL 4.1, as quoted by Apple DTS:
  - Section 6.16.4.4: compare-and-exchange "support atomic types A of atomic_int, atomic_uint, atomic_bool, and
    atomic_float".
  - Section 6.16.4.6, "Atomic Modify Functions (64 Bits)": lists `atomic_max_explicit` and `atomic_min_explicit` for
    `atomic_ulong` in device memory, with `memory_order_relaxed` only. Both return `void`.
  - The Feature Set Tables' footnote 7 ("The full set of 64-bit atomic operations is supported on all platforms
    starting with Apple9") disagrees with the specification. The compiler follows the specification: CAS on
    `atomic_ulong` is "no matching function" under `-std=metal3.1` through `metal4.1`, Xcode 27.0.
  — [Apple Developer Forums thread 848400](https://developer.apple.com/forums/thread/848400)
- Dawn (WebGPU) on Metal: "apple is the most restrictive the atomic64 operations", supporting max/min on u64 only.
  "These operations also do not yield the value. It is a store only" operation, on storage (not workgroup) memory.
  Enabled on Apple9, or on Mac2 + Apple8 ("This means the M2"; not M1). —
  [Dawn commit b620b2fe](https://dawn.googlesource.com/dawn.git/+/b620b2fe514a5ee2e6a2d924fd998c514019da87%5E%21/)
- WebGPU exposes the same subset as `atomic_vec2u_min_max`. The motivating use is the same packing trick:
  software rasterisation with depth and payload in one atomicMax. —
  [Chrome "What's next for WebGPU"](https://developer.chrome.com/blog/next-for-webgpu),
  [Dawn f8bb650b](https://dawn.googlesource.com/dawn/+/f8bb650b474fcec02e883472cf4e31abb5f02dcc)

### Measured (`probe_compile.py`: one Warp `func_native` kernel per call, compiled at runtime by the Metal compiler on the M4 Max)

| MSL call | result |
|---|---|
| `atomic_max_explicit(device atomic_ulong*, ulong, relaxed)` (void) | compiles |
| `atomic_load_explicit` / `atomic_store_explicit` on `atomic_ulong` | no matching function |
| `atomic_exchange_explicit`, `atomic_compare_exchange_weak_explicit` on `atomic_ulong` | no matching function |
| `atomic_fetch_max_explicit`, `atomic_fetch_add_explicit` on `atomic_ulong` | no matching function |
| anything on `atomic_long` | "no type named 'atomic_long'" |

Consequences:
- A 64-bit word can only be *read* atomically as two 32-bit atomic loads (`(device atomic_uint*)p` and `+1`, little
  endian), which is what Newton's HashTable fix does. Such a read can be torn if the word changes between the two
  loads.
- A plain 64-bit load that races with atomics is formally a data race. Mixing 32-bit atomics on one half with 64-bit
  min/max on the whole word is not specified by MSL either; the HashTable fix relies on it, and it measures correct
  (below, and that fix's own test).

### What the Warp fork does with each 64-bit op on Metal (code read, `warp-i2`)
- `warp/native/metal_atomics.h`:
  - `wp_metal_atomic_min/max(unsigned long*)` = a plain pre-read `old = *p`, then `atomic_min/max_explicit`, then
    return `old`. The comment says the returned value "may be stale under contention". The update itself is atomic.
  - 64-bit add/and/or/xor are two 32-bit atomics, each on its own half. The final value is exact; the returned or
    concurrently read value can be torn.
  - 64-bit CAS/exch and signed min/max fall to a generic template that rejects 64-bit types at compile time.
- `warp/_src/codegen.py` `METAL_UNSUPPORTED_ATOMICS_64`:
  - Refuses `atomic_cas` and `atomic_exch` on int64/uint64, and `atomic_min` / `atomic_max` on int64.
  - The kernel compiles with a stand-in and raises at launch on Metal ("Apple GPUs have no 64-bit compare-and-swap
    ...").
  - The check reads the builtin calls of the Python source. So a `wp.atomic_cas(uint64)` that sits in a branch Metal
    never takes still makes the kernel refuse to launch. The CUDA/CPU form must be hidden in a `wp.func_native` body
    under `#if !defined(__METAL_VERSION__)`, as `_hashtable_cas_key` does (Newton 24fb5367).

## Q2. Lock-free techniques without 64-bit CAS, and the pitfalls

1. **Pack and max/min (one-shot publish).** Order the payload so that the integer order of the packed word is the
   order you want the winner chosen in:
   - (score, tie-breaker, id) for contact reduction, already used by Newton's reducer;
   - (float_flip(dist), key_low32) for contact matching, Newton 24fb5367;
   - depth|id in visibility buffers ([Vulcanite](https://github.com/bdwhst/Vulcanite)).

   A 64-bit max/min with no return value is enough when nobody needs to know at once whether they won. Re-reading
   later tells them. It is deterministic if the packed words are distinct, or if ties are equal words.
2. **Claim, then publish.**
   - A 32-bit CAS on the half that distinguishes "empty" moves it to a CLAIMED marker that orders above (for min) or
     below (for max) every valid word.
   - One 64-bit min (or max) then writes the whole word.
   - Readers see empty, claimed (in flight) or settled. The claimant publishes in the same branch, with no loop and
     no dependence on any other thread between claim and publish. This is the Newton HashTable fix (086148e1).
3. **Monotone slots.** If every write to a word in a phase is a max (or every write a min), then:
   - a displaced value never comes back;
   - a stale pre-read (Warp's returned "old") can only under-estimate the slot, which is safe for "did I lose?"
     tests: a stale old >= mine still means I lost;
   - a torn two-half read of a word that held X throughout returns X exactly, and if it changed in between, X is
     gone for good.

   Read-back "slot == mine" decides ownership: a false positive (torn) at worst keeps an unreferenced record, and a
   false negative is impossible. Any non-monotone write (a rollback) breaks this, which is why the designs below
   drop rollbacks on Metal.
4. **Two passes.** When the winner's payload exceeds 64 bits, pack (rank, thread id) into the 64-bit max and let a
   second launch, one thread per destination, copy the winner's payload from a per-thread scratch array. Warp's own
   deterministic counters use the same count-then-re-execute shape
   ([Warp design doc](https://github.com/NVIDIA/warp/blob/main/design/deterministic-execution.md)).
5. **Slot reservation by 32-bit `atomic_add`** (contact buffers, active lists) is native and safe everywhere.

Pitfalls:
- **No forward progress.**
  - Vulkan and Metal say almost nothing about relative progress between workgroups.
  - Apple (and ARM) GPUs measurably do not satisfy the "linear occupancy-bound" model that earlier work hypothesised
    for GPU schedulers. — [Sorensen et al., OOPSLA 2021](https://www.doc.ic.ac.uk/~afd/papers/2021/OOPSLA_GPUs.pdf)
  - Inter-workgroup barriers and locks deadlock under occupancy-bound execution. —
    [Sorensen et al., OOPSLA 2016](https://www.doc.ic.ac.uk/~afd/papers/2016/OOPSLA.pdf)

  A spin lock (`while CAS(lock,0,1)==1`) whose holder sits in a threadgroup that is not scheduled, or in a masked-off
  path of the waiter's own SIMD-group, never returns. On a Mac that means a GPU hang and possibly a WindowServer
  watchdog panic (MPM.md 8c records one).
- **SIMT reconvergence.** "A deadlock-free program on a multiple-instruction multiple-data (MIMD) architecture may
  deadlock on a SIMT machine" because divergent paths are serialised and reconverge at compiler-chosen points
  ([ElTantawy & Aamodt, MICRO 2016](https://people.ece.ubc.ca/aamodt/papers/eltantawy.micro2016.pdf)). NVIDIA
  addressed this in hardware with Volta's independent thread scheduling
  ([Volta Tuning Guide](https://docs.nvidia.com/cuda/archive/11.6.2/pdf/Volta_Tuning_Guide.pdf)). Apple documents
  nothing of the kind; Warp's own Metal guide warns "Spinlocks built from wp.atomic_cas() can hang: Apple GPUs do not
  guarantee forward progress between SIMD lanes" (docs/user_guide/metal.rst).

  The claim/publish wait is weaker than a lock, since the claimant needs only its own next instruction. But it still
  assumes the compiler does not sink the claimant's publish past the loop exit. That is a legal transformation for a
  lane that leaves the loop, and it would let waiting lanes of the same SIMD-group starve it. Hence: **bound every
  wait** and count the failure.
- **ABA.**
  - The claim/publish table never frees slots within a phase (cleared by a separate launch), so a slot goes
    empty -> claimed -> key exactly once. There is no ABA.
  - Rollback-style "restore the previous value if it is still mine" has the ABA problem: the slot can leave "mine"
    and come back. That is another reason not to emulate it with half-word CASes.
- **Memory order.** Only `memory_order_relaxed` exists for device atomics. Ordering between the claim and the publish
  is per-address (same word), which relaxed atomics keep (coherence). Payloads a reader dereferences, such as the
  contact record behind an id, must be written before the id is published *and* read in a later launch, or behind a
  device fence. All sites below either read in a later kernel or read only the word itself.
- **Mixed-size atomics** (32-bit CAS on a half, 64-bit min on the word) are outside the MSL memory model.
  - They are correct on the M4 Max in every probe (P3 below) and in the HashTable test
    (`test_concurrent_distinct_keys`, 262,144 inserts).
  - Treat this as measured hardware behaviour, not a guarantee.

## Q3. Inventory of 64-bit CAS / exch per line (code read; `git grep` on the fork branches)

| site | 1.7 line (`metalsim`, `metalsim-hashtable-metal`; upstream 45458023) | 1.5.2 line (`metalsim-1.5.2`) |
|---|---|---|
| `contact_reduction_global.reduction_rollback_slot` / `reduction_finalize_slot` (uint64 CAS) | present (predictive and two-spatial-depth reducers) | absent (only `reduction_update_slot`, atomic_max) |
| `contact_reduction_global.reduction_try_update_slot` (uses the *returned* uint64 max) | present: atomic on Metal, but the return is a stale pre-read | absent |
| XPBD `build_restitution_manifolds` (int64 CAS on `manifold_key`) | present | absent (no restitution manifolds) |
| Kamino `metrics._atomic_update_metric_max` (int64 `atomic_exch` of argmax keys; 4 int64 argmax arrays) | present, generic helper | present, inline at 5 sites (`r_eom`, `r_kinematics` x2, `r_cts_joints`, `r_cts_limits`) |
| HashTable 64-bit CAS | fixed (`_hashtable_cas_key` in func_native, claim/publish on Metal) | fixed (same) |
| 32-bit only (no change needed): Kamino `rcm_batch` level CAS (int32), `fk` max_residual CAS loop (float32), contact-overflow flags (int32), VBD `particle_contact_head` exch (int32), XPBD `manifold_head` exch (int32) | | |
| **Not 64-bit, but a true spin lock:** `geometry/raycast.py` `_spinlock_acquire` (`while atomic_cas(lock,0,0,1)==1`) | present | present |

Measured failure inventory on Metal (Warp `warp-i2`, Newton `metalsim-hashtable-metal` tree, Newton's tests with the
device list forced to "all"; `tests_metal.log`):
- **test_contact_reduction_global**: 3 failed, 25 passed. The three are `test_centered_two_spatial_depths_*`
  (`prefers_inner_then_outer`, `voxel_only_claim`, `full_buffer_does_not_publish_voxel_key`). The refused kernels are
  test-local kernels (`store_two_depth_contact_kernel`, `reduce_kernel`, `exhaust_then_reduce_kernel`) that call
  `export_and_reduce_contact_centered_two_spatial_depths`. That function reaches `reduction_rollback_slot` /
  `reduction_finalize_slot`: `atomic_cas(uint64)`. The tests do not call CAS themselves.
- **test_solver_xpbd**: 12 failed.
  - 11 refuse `build_restitution_manifolds: atomic_cas(int64)`.
  - 1, `test_xpbd_parent_f_consistent_across_solvers`, is unrelated: MuJoCo Warp `contact_force_fn` signature
    mismatch in `convert_mjw_contacts_to_newton_kernel`, an environment matter.
- **kamino/test_kamino_solvers_metrics**: 15 failed, 6 passed on the default device metal:0. The first failure
  inspected refuses `_compute_cts_joints_residual: atomic_exch(int64)`; the other 14 were not inspected one by one.
- **test_hashtable**: all pass.
- With stock 45458023 (what `.venv` has installed today), every HashTable user also fails on Metal: 15 of 28 in
  test_contact_reduction_global.

## Q4. Per-site designs

Common scaffolding for Newton, copied from the HashTable fix (086148e1 / 24fb5367):
- a `_use_metal_protocol()` `func_native` that returns `true` under `__METAL_VERSION__`;
- `_load_u64(arr, idx)`: two relaxed 32-bit atomic loads on Metal, a plain load elsewhere;
- the CUDA/CPU 64-bit CAS kept inside a `func_native` `#else` branch, so the Python codegen never sees a
  `wp.atomic_cas(int64/uint64)` and the kernel is not refused on Metal.

CPU and CUDA execute exactly the original code.

### 4a. contact_reduction_global (1.7 line only)

What the CAS does:
- A candidate first claims a slot provisionally: `try_update` = `atomic_max` of a packed value with **contact id 0**.
  ID 0 is reserved: export kernels skip it (`if contact_id == 0: continue`), and `_pair_duplicate_bit` compares
  buffer record 0.
- Only a candidate that still holds at least one slot allocates a buffer record.
- **finalize** = `CAS(slot, provisional -> provisional | id)` succeeds only if nobody displaced the provisional.
- **rollback** = `CAS(slot, provisional -> previous)` when allocation fails (buffer exhausted): it restores the
  displaced winner unless someone newer displaced the provisional.

Metal protocol:

```python
@wp.func
def reduction_finalize_slot(entry_idx, slot_id, provisional_value, final_value, values, capacity) -> bool:
    value_idx = slot_id * capacity + entry_idx
    if _use_metal_protocol():
        wp.atomic_max(values, value_idx, final_value)            # hardware 64-bit max (Apple9+)
        return _load_u64(values, value_idx) == final_value       # two 32-bit atomic loads
    return _cas_u64(values, value_idx, provisional_value, final_value) == provisional_value   # CUDA/CPU, func_native

@wp.func
def reduction_rollback_slot(entry_idx, slot_id, provisional_value, previous_value, values, capacity):
    if _use_metal_protocol():
        return      # keep slots monotone: the id-0 provisional stays, readers skip it
    _cas_u64(values, slot_id * capacity + entry_idx, provisional_value, previous_value)
```

Optionally, on Metal, the decision reads (`still_wins`, `clearance_won`, the `try_update` pre-read) use `_load_u64`
instead of a plain racy 64-bit load.

Why it is correct:
- `final = provisional | id`, with id in the lowest bits (32 fast, 20 deterministic) and `provisional`'s id bits zero.
- If the slot still holds our provisional, the max writes `final`: success, as with the CAS.
- If the slot holds `V > final`, the max does nothing and the read-back differs: failure, as with the CAS.
- Only `provisional < V < final` behaves differently: V is a materialised value of a contact with the **identical**
  (score, fingerprint) and a smaller id. Metal then installs ours where the CAS would have failed. Both outcomes are
  orderings the reducer's own max rule allows (an existing-id update would do the same). The displaced twin record
  stays allocated but unreferenced, which is wasted buffer only.
- Since no Metal write lowers a slot, ownership by read-back is sound (Q2 item 3). A reclaimed id is never referenced
  later, because a displaced final cannot come back.
- Two twins with equal provisionals that both think they won (stale pre-read) both finalize. The larger id stays,
  the other reads back a mismatch and reclaims.
- No thread waits on another; there is no loop.

Behaviour difference: rollback only runs when `export_contact_to_buffer` fails, i.e. the contact buffer and
reclamation are exhausted, already a reported overflow. On Metal the slot then keeps an id-0 provisional. It exports
nothing and blocks lower-scoring candidates for that slot in that frame, where CUDA would restore the earlier winner.
So Metal can export fewer contacts **only after an overflow**.

Tests:
- `full_buffer_does_not_publish_voxel_key` asserts `get_winning_contacts(reducer) == []`. With rollback skipped, the
  helper sees the provisional id-0 words. The helper should skip id 0, as the export kernel does:
  `if val != 0 and (val & mask) != 0:`. This is neutral on CPU/CUDA.
- The other two tests run single-thread sequences; the protocol gives the same answers there (inferred from the code
  paths, not yet run).

### 4b. XPBD restitution manifold table (1.7 line only)

What the CAS does:
- An open-addressing insert of `key = (lo+1)*(body_count+1) + (hi+1) >= 1`, with empty = 0 (the table is zeroed per
  substep).
- `prev == 0 or prev == key`: the slot is ours or shared. The contact is pushed on the slot's chain with 32-bit
  `atomic_add` / `atomic_exch` (native).
- Readers (`select_manifold_contacts`, `solve_manifold_restitution`) only test `manifold_key[tid] == 0`.

Why not just 32 bits: the key exceeds 32 bits beyond about 65k bodies (4,096 G1 worlds is about 120k).

Metal protocol, the HashTable claim/publish with a tag so that empty can stay 0:

```python
_TAG = wp.constant(wp.int64(1 << 62))     # stored = key | TAG, key < (body_count+1)^2 <= 2^62
# Metal helpers (func_native, as in hashtable.py): load_hi, load_key (2x32 atomic loads),
# claim (32-bit CAS on the high word 0 -> 0x80000000), publish (atomic_min_explicit on the atomic_ulong view)
if _use_metal_protocol():
    stored = key | _TAG
    found = False; passes = int(0); probes = int(0); pending = True
    while pending and passes < _MAX_PASSES:      # e.g. 1 << 16; every lane runs the loop
        passes += 1
        hi = load_hi(manifold_key, slot)
        if hi == wp.uint32(0):
            if claim(manifold_key, slot):        # (0x80000000, 0) > stored, so the min writes stored in one atomic
                publish(manifold_key, slot, stored); found = True; pending = False
        elif hi != wp.uint32(0x80000000):
            if load_key(manifold_key, slot) == stored: found = True; pending = False
            else:
                slot = (slot + 1) % table_size; probes += 1
                if probes >= table_size: pending = False
        # else: claimed, re-read next pass
    if found: (unchanged chain push)
    else: wp.atomic_add(restitution_insert_failures, 0, 1)   # new diagnostic; no restitution for this contact
else:
    (original loop, with the int64 CAS moved into a func_native #else body)
```

- The probe sequence starts at `key % table_size` from the untagged key, so it is identical on every device.
- The tag keeps occupied high words in [0x40000000, 0x7FFFFFFF], disjoint from empty (0) and claimed (0x80000000).
- Readers are unchanged, since they only test `!= 0`.
- The physics is order-independent: selection sorts and breaks ties to lower index, as the docstring states. So
  slot placement differing between runs does not change results (it already differs on CUDA).
- The only wait is on a claimed slot, whose claimant needs one more instruction. The bound turns a pathological
  scheduler or compiler into a counted failure instead of a hang.

### 4c. Kamino metrics argmax (1.7 helper; 1.5.2 inline sites)

What the code does:
1. The per-world float metric max is a 32-bit CAS loop. That loop is lock-free, not a lock: some lane's CAS always
   succeeds.
2. Then, if `value >= previous_max`, `atomic_exch(argmax[wid], key)` with a 63-bit `build_pair_key2` key. NaN sets
   the metric to NaN (sticky) and the key to -1.

Even on CUDA this is racy: the max and the key are separate atomics. A thread with 5 can exchange after a thread
with 7, so the stored key need not belong to the maximum, and the result is run-to-run nondeterministic. A key has
63 bits, so key and value cannot share one 64-bit word.

Metal protocol, deterministic and an exact argmax:
- **Per thread**: `key_scratch[tid] = key` (own slot, plain store).
- **Per thread**: `wp.atomic_max(argmax_packed[wid], (rank << 32) | (0xFFFFFFFF - tid))`, where
  `argmax_packed: uint64[num_worlds]` is zeroed with the metrics. The rank is `float_flip(value)` for a finite value
  and `0xFFFFFFFF` for NaN, which ranks above every finite float_flip, matching NaN's stickiness.
- **Finalize launch**, dim = num_worlds, after the metric kernels: if `p == 0`, write -1. Otherwise, if
  `p >> 32 == 0xFFFFFFFF` (NaN), write -1. Otherwise write
  `argmax[wid] = key_scratch[0xFFFFFFFF - (p & 0xFFFFFFFF)]`.
- One `key_scratch` sized to the largest metric launch serves every metric if each metric has its own
  `argmax_packed` row, or if the metric launches each get their own finalize.

The metric max itself is unchanged. The packed word is a single native atomic and nothing waits. The result is the
key of the thread holding the maximum, lowest thread id on exact ties. CUDA/CPU keep the exchange.

Whether test_kamino_solvers_metrics expects a particular key on ties has not been checked. Its docstrings for the
segment helpers keep "the earliest index", which the lowest-tid rule matches within a launch.

Cheaper but rejected: two 32-bit exchanges of the key halves give torn keys (a pair that never existed) under
contention.

### 4d. test_contact_reduction_global's three errors

Covered by 4a: the test kernels call `export_and_reduce_contact_centered_two_spatial_depths`, not CAS. With 4a,
kernels launch on Metal, and only the `get_winning_contacts` helper needs to skip id 0 (4a, Tests).

## Q5. Prototypes on the M4 Max (measured, `probe_atomics64.py`; each case repeated 3-5 times)

| probe | setup | result |
|---|---|---|
| P1 uint64 `atomic_max` / `atomic_min` under contention | 2^20 random words into 4 slots | final values exact, 5/5 runs |
| P2 Warp refusal | `atomic_cas(uint64)`, `atomic_exch(int64)`, `atomic_min(int64)` | each raises at launch on Metal; on CPU they run |
| P3 claim/publish insert, empty = 0, tag 2^62, bounded at 65,536 passes | 4,096 threads / 64 keys; 262,144 / 8,192; 1,048,576 / 65,536; table = 2x keys | every key in exactly one slot, every thread got its key's slot, **0 bounded-loop exits**, 3/3 runs each |
| P4 packed `(value bits, ~tid)` argmax + finalize | 256 worlds x 4,096 threads, 5,000 forced ties | equals NumPy argmax (lowest index on ties) 3/3 |
| P5 provisional (id 0) -> final by `atomic_max` + read-back, no rollback | 262,144 threads, 64 slots | every slot holds the max score with an owner that kept its record; 186 kept, 3/3 |

Not covered by the probes:
- the Newton code itself (the designs are not implemented there by this note);
- runs under other GPU load;
- the compiler-sinking hazard. It did not occur here (0 stalls), which is evidence for this compiler build only.

## Open questions

- Implement 4a-4c in the fork branches and run Newton's contact reduction, XPBD and Kamino metrics tests on Metal and
  CPU under a watchdog.
- The raycast spin lock is a real lock (32-bit, so not refused). It is unsafe on Metal for the reasons in Q2 and
  should become a claim-free formulation, e.g. a packed `atomic_min` of (distance, index) if it guards a closest-hit
  update. Not examined here.
- Whether Apple will ship 64-bit CAS: DTS suggests filing an enhancement request; nothing is promised
  ([forum thread](https://developer.apple.com/forums/thread/848400)).

## Implementation (2026-10-12)

The designs of Q4 are in the Newton fork (pulipakaa24/newton), CPU and CUDA code unchanged (the 64-bit CAS / exchange
moved into `func_native` helpers with a Metal `#if` branch):

| site | 1.7 line (`metalsim`, `metalsim-hashtable-metal`) | 1.5.2 line (`metalsim-1.5.2`) |
|---|---|---|
| contact_reduction_global finalize / rollback (4a) | `852a9653` / `ffae2416` | absent |
| XPBD restitution manifold table (4b) | `210e7768` / `391650b6` | absent |
| Kamino metric argmax keys (4c) | `f6c8f60d` / `4923e0f3` | `3813cbba` (the 5 inline sites through one helper) |

Departures from the designs above:
- 4b: a contact that runs out of passes (bound: table size + 65,536 passes) is left out of restitution, as on CUDA
  when the table is full, and counted in `SolverXPBD.restitution_insert_failures`; the HashTable's claim loop got the
  same bound (returns -1, counted by the reducers in `ht_insert_failures`): `7dd71ed9` / `a4be3604` (1.7),
  `e0c7c82e` (1.5.2).
- 4c zeroes the packed maxima before each metric launch and finalizes right after it, so one packed row and one key
  scratch serve every metric; the finalize stores the key only when the launch's maximum is the world's maximum after
  the launch (the exchange's own condition), so evaluations without a reset in between keep the exchange's semantics.
  On the 1.5.2 line (no NaN handling there) NaN values do not take part, as their `value >= previous_max` never held.

Tests (Warp fork `152808b7` built, devices cpu + metal:0, Newton's test files under pytest with the device mode "all";
logs `runs/rigid17/out/`): on `metalsim-hashtable-metal` (what `.venv` installs) 98 failures -> 13 over
test_contact_reduction_global, test_contact_reduction, test_hashtable, test_contact_matching, test_collision_pipeline,
test_narrow_phase, test_sdf_contact, test_heightfield, test_mesh_backface, test_rigid_contact, test_solver_xpbd and
kamino/test_kamino_solvers_metrics; no test went from pass to fail; the 13 left are SolverMuJoCo tests that fail to
build `convert_mjw_contacts_to_newton_kernel` against the MuJoCo Warp fork 3.14 (`contact_force_fn` signature,
environment). `metalsim` gives identical outcomes. 1.5.2 Kamino metrics: CPU 9/9; Metal 8/9 (before: 5 errors, the
refused exchange); the Metal failure is test_06's dense-vs-sparse `f_ncp` comparison, 2.1e-6 relative against rtol
1e-6 in a kernel these changes do not touch. The raycast spin lock (Q3) is unchanged and remains a hang risk on Metal.
