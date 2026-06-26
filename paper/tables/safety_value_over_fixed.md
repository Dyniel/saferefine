# Safety Value Over Fixed Refinement

| setting | fixed_gain | fixed_harm | selected_policy | policy_gain | policy_harm | harm_prevented | harm_reduction | worst_drop_improvement | gain_delta_vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC / UNet | +0.0026 | 0.0034 | zoo CRC | +0.0033 | 0.0015 | +0.0019 | 55.9% | +0.3709 | +0.0007 |
| Kvasir / GraphSeg | +0.0073 | 0.0424 | baseline CRC | +0.0095 | 0.0016 | +0.0408 | 96.2% | +0.3375 | +0.0022 |
| Kvasir / UNet | +0.0078 | 0.0082 | baseline CRC | +0.0000 | 0.0000 | +0.0082 | 100.0% | +0.3400 | -0.0078 |
| MSD Heart MRI / GraphSeg | +0.0211 | 0.0499 | zoo CRC | +0.0000 | 0.0000 | +0.0499 | 100.0% | +0.3263 | -0.0211 |
| MSD Heart MRI / UNet | +0.0017 | 0.0070 | zoo CRC | +0.0000 | 0.0000 | +0.0070 | 100.0% | +0.1641 | -0.0017 |
| PH2 / UNet | +0.0053 | 0.0000 | baseline CRC | +0.0000 | 0.0000 | +0.0000 | n/a | +0.0087 | -0.0053 |
| Polyp ext. / GraphSeg | -0.0211 | 0.0524 | zoo CRC | +0.0026 | 0.0000 | +0.0524 | 100.0% | +0.4638 | +0.0237 |
| Polyp ext. / UNet | -0.0017 | 0.0267 | baseline CRC | +0.0037 | 0.0018 | +0.0249 | 93.3% | +0.3401 | +0.0054 |
