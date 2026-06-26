# Cross-Dataset Policy Transfer

| transfer | best_practical_risk | target_gain | target_harm | target_worst_drop | target_revert_rate | selected_action | strict_exact_fallback | paper_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isic_to_ph2_graphseg | changed | +0.0063 | 0.0000 | -0.0018 | 0.067 | binary_morph:a11.00 | yes | positive dermoscopy transfer |
| isic_to_ph2_unet | change_plus_geom | +0.0000 | 0.0000 | +0.0000 | 1.000 | binary_morph:a5.00 | yes | positive dermoscopy transfer |
| kvasir_to_polyps_graphseg | changed | -0.0152 | 0.0239 | -0.4624 | 0.426 | binary_morph:a3.00 | yes | external stress test; practical transfer can be unsafe |
| kvasir_to_polyps_unet | change_plus_geom | +0.0000 | 0.0000 | -0.0016 | 0.990 | binary_morph:a7.00 | yes | external stress test; high fallback limits harm |
| kvasir_to_sessile_graphseg | change_plus_geom | +0.0011 | 0.0000 | -0.0106 | 0.500 | binary_morph:a7.00 | yes | related endoscopy transfer; mixed utility, fallback important |
| kvasir_to_sessile_unet | change_plus_geom | +0.0000 | 0.0000 | +0.0000 | 1.000 | binary_morph:a7.00 | yes | related endoscopy transfer; mixed utility, fallback important |
