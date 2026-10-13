#### twoway (16 perturbed L4 runs; base reruns bitwise: {'base2': False})

| Metal run | frame | M | floor p50 / p90 / max | pairwise p90 / p95 / max | rank in floor / pairwise (%) | M / max floor | M / p95 pairwise |
|---|---|---|---|---|---|---|---|
| metal_floors_integrated | 2 | 0.00311 um | 0.0569 um / 0.109 um / 0.134 um | 0.134 um / 0.144 um / 0.153 um | 25 / 8 | 0.02 | 0.02 |
| metal_floors_integrated | 52 | 5.87 um | 14.6 um / 18.2 um / 22.7 um | 21.5 um / 22.7 um / 23.9 um | 12 / 4 | 0.26 | 0.26 |
| metal_floors_integrated | 100 | 206 um | 163 um / 197 um / 204 um | 189 um / 197 um / 219 um | 100 / 97 | 1.01 | 1.04 |
| metal_floors_integrated | worst frame | | | | | 1.01 | 1.04 |
| metal_floors_integrated_nofastmath | 2 | 0.00308 um | 0.0569 um / 0.109 um / 0.134 um | 0.134 um / 0.144 um / 0.153 um | 25 / 8 | 0.02 | 0.02 |
| metal_floors_integrated_nofastmath | 52 | 5.07 um | 14.6 um / 18.2 um / 22.7 um | 21.5 um / 22.7 um / 23.9 um | 12 / 4 | 0.22 | 0.22 |
| metal_floors_integrated_nofastmath | 100 | 125 um | 163 um / 197 um / 204 um | 189 um / 197 um / 219 um | 19 / 35 | 0.61 | 0.63 |
| metal_floors_integrated_nofastmath | worst frame | | | | | 0.83 | 0.80 |

Verdicts: metal_floors_integrated: OUTSIDE max floor, OUTSIDE p95 pairwise; metal_floors_integrated_nofastmath: inside max floor, inside p95 pairwise

#### multi_material (16 perturbed L4 runs; base reruns bitwise: {'base2': True})

| Metal run | frame | M | floor p50 / p90 / max | pairwise p90 / p95 / max | rank in floor / pairwise (%) | M / max floor | M / p95 pairwise |
|---|---|---|---|---|---|---|---|
| metal_floors_integrated | 6 | 0.0654 um | 0.0784 um / 0.0842 um / 0.0856 um | 0.108 um / 0.109 um / 0.111 um | 38 / 15 | 0.76 | 0.60 |
| metal_floors_integrated | 36 | 17.6 mm | 13.7 mm / 15.2 mm / 15.6 mm | 16 mm / 16.5 mm / 17.5 mm | 100 / 100 | 1.13 | 1.07 |
| metal_floors_integrated | 60 | 29.4 mm | 23.8 mm / 25.9 mm / 28.6 mm | 28.8 mm / 30.5 mm / 32 mm | 100 / 92 | 1.03 | 0.96 |
| metal_floors_integrated | worst frame | | | | | 1.13 | 1.07 |
| metal_floors_integrated_nofastmath | 6 | 0.0639 um | 0.0784 um / 0.0842 um / 0.0856 um | 0.108 um / 0.109 um / 0.111 um | 38 / 15 | 0.75 | 0.59 |
| metal_floors_integrated_nofastmath | 36 | 17.3 mm | 13.7 mm / 15.2 mm / 15.6 mm | 16 mm / 16.5 mm / 17.5 mm | 100 / 98 | 1.11 | 1.05 |
| metal_floors_integrated_nofastmath | 60 | 29.4 mm | 23.8 mm / 25.9 mm / 28.6 mm | 28.8 mm / 30.5 mm / 32 mm | 100 / 92 | 1.03 | 0.96 |
| metal_floors_integrated_nofastmath | worst frame | | | | | 1.15 | 1.11 |

Verdicts: metal_floors_integrated: OUTSIDE max floor, OUTSIDE p95 pairwise; metal_floors_integrated_nofastmath: OUTSIDE max floor, OUTSIDE p95 pairwise

#### granular (16 perturbed L4 runs; base reruns bitwise: {'base2': True})

| Metal run | frame | M | floor p50 / p90 / max | pairwise p90 / p95 / max | rank in floor / pairwise (%) | M / max floor | M / p95 pairwise |
|---|---|---|---|---|---|---|---|
| metal_floors_integrated | 6 | 244 um | 2.57 um / 372 um / 372 um | 372 um / 372 um / 372 um | 81 / 69 | 0.66 | 0.66 |
| metal_floors_integrated | 36 | 8.38 mm | 442 um / 17.5 mm / 17.5 mm | 17.5 mm / 21.1 mm / 21.1 mm | 62 / 45 | 0.48 | 0.40 |
| metal_floors_integrated | 60 | 17.4 mm | 1.42 mm / 43.1 mm / 43.1 mm | 43.1 mm / 55.8 mm / 55.8 mm | 62 / 45 | 0.40 | 0.31 |
| metal_floors_integrated | worst frame | | | | | 0.66 | 0.66 |
| metal_floors_integrated_nofastmath | 6 | 244 um | 2.57 um / 372 um / 372 um | 372 um / 372 um / 372 um | 81 / 69 | 0.66 | 0.66 |
| metal_floors_integrated_nofastmath | 36 | 8.38 mm | 442 um / 17.5 mm / 17.5 mm | 17.5 mm / 21.1 mm / 21.1 mm | 62 / 45 | 0.48 | 0.40 |
| metal_floors_integrated_nofastmath | 60 | 17.4 mm | 1.42 mm / 43.1 mm / 43.1 mm | 43.1 mm / 55.8 mm / 55.8 mm | 62 / 45 | 0.40 | 0.31 |
| metal_floors_integrated_nofastmath | worst frame | | | | | 0.66 | 0.66 |

Verdicts: metal_floors_integrated: inside max floor, inside p95 pairwise; metal_floors_integrated_nofastmath: inside max floor, inside p95 pairwise

#### anymal (16 perturbed L4 runs; base reruns bitwise: {'base2': True})

| Metal run | frame | M | floor p50 / p90 / max | pairwise p90 / p95 / max | rank in floor / pairwise (%) | M / max floor | M / p95 pairwise |
|---|---|---|---|---|---|---|---|
| metal_floors_integrated | 2 | 0.919 um | 0.187 um / 1.89 um / 1.9 um | 1.9 um / 1.91 um / 1.92 um | 81 / 69 | 0.48 | 0.48 |
| metal_floors_integrated | 26 | 90.9 um | 13.1 um / 92.9 um / 96.4 um | 96.2 um / 96.4 um / 96.6 um | 88 / 84 | 0.94 | 0.94 |
| metal_floors_integrated | 50 | 358 um | 137 um / 346 um / 355 um | 349 um / 355 um / 357 um | 100 / 100 | 1.01 | 1.01 |
| metal_floors_integrated | worst frame | | | | | 1.22 | 1.22 |
| metal_floors_integrated_nofastmath | 2 | 0.919 um | 0.187 um / 1.89 um / 1.9 um | 1.9 um / 1.91 um / 1.92 um | 81 / 69 | 0.48 | 0.48 |
| metal_floors_integrated_nofastmath | 26 | 90.9 um | 13.1 um / 92.9 um / 96.4 um | 96.2 um / 96.4 um / 96.6 um | 88 / 84 | 0.94 | 0.94 |
| metal_floors_integrated_nofastmath | 50 | 351 um | 137 um / 346 um / 355 um | 349 um / 355 um / 357 um | 88 / 94 | 0.99 | 0.99 |
| metal_floors_integrated_nofastmath | worst frame | | | | | 1.22 | 1.22 |

Verdicts: metal_floors_integrated: OUTSIDE max floor, OUTSIDE p95 pairwise; metal_floors_integrated_nofastmath: inside max floor, inside p95 pairwise

#### column_a2 (16 perturbed L4 runs; base reruns bitwise: {'base2': True})

| Metal run | frame | M | floor p50 / p90 / max | pairwise p90 / p95 / max | rank in floor / pairwise (%) | M / max floor | M / p95 pairwise |
|---|---|---|---|---|---|---|---|
| metal_floors_integrated | 1 | 0.00305 um | 0.0193 um / 23.1 um / 23.1 um | 23.1 um / 23.1 um / 23.1 um | 25 / 8 | 0.00 | 0.00 |
| metal_floors_integrated | 91 | 0.163 um | 0.173 um / 1.68 mm / 1.68 mm | 1.68 mm / 1.68 mm / 1.68 mm | 25 / 10 | 0.00 | 0.00 |
| metal_floors_integrated | 180 | 0.163 um | 0.178 um / 1.68 mm / 1.68 mm | 1.68 mm / 1.68 mm / 1.68 mm | 25 / 8 | 0.00 | 0.00 |
| metal_floors_integrated | worst frame | | | | | 0.00 | 0.00 |
| metal_floors_integrated_nofastmath | 1 | 0.00338 um | 0.0193 um / 23.1 um / 23.1 um | 23.1 um / 23.1 um / 23.1 um | 25 / 8 | 0.00 | 0.00 |
| metal_floors_integrated_nofastmath | 91 | 0.162 um | 0.173 um / 1.68 mm / 1.68 mm | 1.68 mm / 1.68 mm / 1.68 mm | 25 / 9 | 0.00 | 0.00 |
| metal_floors_integrated_nofastmath | 180 | 0.165 um | 0.178 um / 1.68 mm / 1.68 mm | 1.68 mm / 1.68 mm / 1.68 mm | 25 / 8 | 0.00 | 0.00 |
| metal_floors_integrated_nofastmath | worst frame | | | | | 0.00 | 0.00 |

Verdicts: metal_floors_integrated: inside max floor, inside p95 pairwise; metal_floors_integrated_nofastmath: inside max floor, inside p95 pairwise

