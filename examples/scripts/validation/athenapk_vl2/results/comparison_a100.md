| case | cycles | AthenaPK GPU vs CPU | astronomix native vs AthenaPK CPU | astronomix native vs AthenaPK GPU | astronomix pallas vs AthenaPK CPU | astronomix pallas vs AthenaPK GPU |
|---|---|---|---|---|---|---|
| cp_alfven_3d | 205 | 5.3e-14 | 6.4e-14 | 6.2e-14 | 6.8e-14 | 6.5e-14 |
| fast_wave_3d | 36 | 3.1e-12 | 3.8e-12 | 3.5e-12 | 3.8e-12 | 3.5e-12 |
| slow_wave_3d | 36 | 3.0e-12 | 3.9e-12 | 3.7e-12 | 3.9e-12 | 3.7e-12 |
| entropy_wave_3d_advected | 60 | 2.8e-16 | 2.4e-15 | 2.4e-15 | 2.4e-15 | 2.4e-15 |
| orszag_tang | 404 | 6.5e-14 | 7.1e-14 | 7.0e-14 | 7.1e-14 | 7.0e-14 |
| orszag_tang_fofc | 404 | 6.5e-14 | 7.1e-14 | 7.0e-14 | 7.1e-14 | 7.0e-14 |
| orszag_tang_extended_glm | 404 | 6.3e-14 | 7.5e-14 | 8.5e-14 | 7.5e-14 | 8.5e-14 |
| orszag_tang_donor_cell | 383 | 9.3e-15 | 9.2e-15 | 1.0e-14 | 9.2e-15 | 1.0e-14 |
| field_loop_hlle | 703 | 1.0e-13 | 2.0e-13 | 2.0e-13 | 2.0e-13 | 2.0e-13 |
| magnetized_blast_3d_fofc | 89 | 2.0e-13 | 2.0e-13 | 2.0e-13 | 2.0e-13 | 2.0e-13 |
| mhd_rotor_outflow | 168 | 9.2e-12 | 1.1e-11 | 9.3e-12 | 1.1e-11 | 9.3e-12 |
| brio_wu_outflow | 390 | 5.8e-15 | 6.6e-15 | 5.1e-15 | 6.6e-15 | 5.1e-15 |
| sod_hllc | 224 | 1.2e-15 | 1.8e-15 | 2.3e-15 | 1.8e-15 | 2.2e-15 |
| sod_hlle | 224 | 7.4e-16 | 1.6e-15 | 1.8e-15 | 1.6e-15 | 1.8e-15 |
| sound_wave_3d_hlle | 36 | 2.1e-12 | 2.7e-12 | 2.5e-12 | 2.7e-12 | 2.6e-12 |
| sedov_3d_hllc_fofc_floors | 58 | 5.4e-16 | 9.2e-16 | 8.5e-16 | 8.4e-16 | 8.7e-16 |
| einfeldt_mhd_fofc | 260 / 262 | 5.2e-03 | 5.2e-03 | 2.9e-12 | 5.2e-03 | 2.9e-12 |
| colliding_flows_mhd_fofc | 100 | 5.7e-02 | 5.7e-02 | 6.7e-11 | 5.7e-02 | 6.7e-11 |
| low_beta_blast_fofc_floors | 24 | 1.6e-01 | 1.4e-01 | 5.7e-02 | 1.4e-01 | 5.7e-02 |
