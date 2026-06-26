# Per-Image Action Selection

| setting | policy | gain | harm | harmed_rate | drop_gt_0.05 | cvar_harm_10 | worst_drop | revert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| ISIC / UNet | oracle | +0.0111 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0001 | 0.000 |
| ISIC / UNet | per_image_changed | +0.0000 | 0.0000 | 0.024 | 0.000 | 0.0000 | -0.0001 | 0.342 |
| ISIC / UNet | per_image_geom | +0.0000 | 0.0000 | 0.100 | 0.000 | 0.0002 | -0.0010 | 0.300 |
| ISIC / UNet | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.342 |
| ISIC / UNet | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| ISIC / UNet | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / GraphSeg | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / GraphSeg | oracle | +0.0615 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.000 |
| Kvasir / GraphSeg | per_image_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / GraphSeg | per_image_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / GraphSeg | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / GraphSeg | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / GraphSeg | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / UNet | oracle | +0.0210 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0002 | 0.000 |
| Kvasir / UNet | per_image_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / UNet | per_image_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / UNet | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / UNet | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Kvasir / UNet | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / GraphSeg | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / GraphSeg | oracle | +0.1237 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.032 |
| MSD Heart MRI / GraphSeg | per_image_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / GraphSeg | per_image_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / GraphSeg | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / GraphSeg | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / GraphSeg | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / UNet | oracle | +0.0484 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.081 |
| MSD Heart MRI / UNet | per_image_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / UNet | per_image_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / UNet | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / UNet | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| MSD Heart MRI / UNet | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| PH2 / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| PH2 / UNet | oracle | +0.0064 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0003 | 0.000 |
| PH2 / UNet | per_image_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| PH2 / UNet | per_image_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| PH2 / UNet | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| PH2 / UNet | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| PH2 / UNet | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / GraphSeg | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / GraphSeg | oracle | +0.0656 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.015 |
| Polyp ext. / GraphSeg | per_image_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / GraphSeg | per_image_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / GraphSeg | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / GraphSeg | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / GraphSeg | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / UNet | oracle | +0.0289 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.068 |
| Polyp ext. / UNet | per_image_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / UNet | per_image_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / UNet | per_image_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / UNet | per_image_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
| Polyp ext. / UNet | per_image_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |
