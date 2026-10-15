
### g1 (4096 worlds)

| cache | workers | runs | process wall [s] | build [s] | first step [s] | modules compiled | sum of module load timers [s] | speed-up vs serial |
|---|---|---|---|---|---|---|---|---|
| cold_prime | w0 | 1 | 15.9 | 14.4 | 0.1 | 43 | 13.8 | 1.00x |
| cold | w0 | 2 | 8.2 | 6.7 | 0.1 | 43 | 6.1 | 1.00x |
| cold | w1 | 2 | 8.3 | 6.8 | 0.1 | 43 | 6.2 | 0.99x |
| cold | w2 | 2 | 8.1 | 6.6 | 0.1 | 43 | 12.1 | 1.01x |
| cold | w4 | 2 | 8.0 | 6.5 | 0.1 | 43 | 23.7 | 1.03x |
| cold | w8 | 2 | 7.9 | 6.4 | 0.1 | 43 | 43.1 | 1.04x |
| warm | w0 | 2 | 3.8 | 2.3 | 0.1 | 0 | 1.8 | 1.00x |
| warm | w2 | 2 | 3.7 | 2.2 | 0.1 | 0 | 3.3 | 1.04x |
| warm | w4 | 2 | 3.6 | 2.1 | 0.1 | 0 | 6.2 | 1.05x |
| warm | w8 | 2 | 3.6 | 2.1 | 0.1 | 0 | 11.6 | 1.06x |
| cold_det | w0 | 1 | 39.8 | 37.6 | 0.2 | 47 | 27.3 | 1.00x |
| cold_det | w8 | 1 | 9.5 | 7.3 | 0.2 | 47 | 50.2 | 4.19x |

Kernel cache files compared with the serial build: 9 runs, 782 files, 0 differ.
Distinct result hashes: default mode: 19, deterministic: 1

### box_sand (2 worlds)

| cache | workers | runs | process wall [s] | build [s] | first step [s] | modules compiled | sum of module load timers [s] | speed-up vs serial |
|---|---|---|---|---|---|---|---|---|
| cold_prime | w0 | 1 | 16.2 | 8.5 | 6.6 | 113 | 12.9 | 1.00x |
| cold | w0 | 2 | 16.2 | 8.5 | 6.5 | 113 | 12.8 | 1.00x |
| cold | w1 | 2 | 16.1 | 8.5 | 6.5 | 113 | 12.8 | 1.00x |
| cold | w2 | 2 | 16.0 | 8.3 | 6.5 | 113 | 18.6 | 1.01x |
| cold | w4 | 2 | 15.8 | 8.2 | 6.5 | 113 | 29.4 | 1.02x |
| cold | w8 | 2 | 15.8 | 8.2 | 6.4 | 113 | 47.5 | 1.02x |
| warm | w0 | 2 | 7.9 | 4.0 | 2.8 | 0 | 4.6 | 1.00x |
| warm | w2 | 2 | 7.7 | 3.8 | 2.7 | 0 | 6.4 | 1.03x |
| warm | w4 | 2 | 7.6 | 3.8 | 2.7 | 0 | 9.5 | 1.04x |
| warm | w8 | 2 | 7.6 | 3.8 | 2.7 | 0 | 15.2 | 1.04x |

Kernel cache files compared with the serial build: 8 runs, 1808 files, 0 differ.
Distinct result hashes: default mode: 1
