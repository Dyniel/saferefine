# Decision Baseline Policy Comparison

| setting | policy | gain | harm | harmed_rate | drop_gt_0.05 | cvar_harm_10 | worst_drop | revert_rate | action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| ISIC / UNet | best_fixed_cal_utility | +0.0008 | 0.0003 | 0.364 | 0.000 | 0.0017 | -0.0065 | 0.000 | isic2018_task1_mediafinal_unet_e120_zoo:prob_gaussian:a2.00 |
| ISIC / UNet | oracle | +0.0114 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.000 |  |
| ISIC / UNet | crc_changed | +0.0033 | 0.0015 | 0.218 | 0.008 | 0.0148 | -0.1519 | 0.032 | isic2018_task1_mediafinal_unet_e120_zoo:largest_cc:a0.00 |
| ISIC / UNet | crc_geom | +0.0030 | 0.0009 | 0.158 | 0.006 | 0.0094 | -0.1519 | 0.152 | isic2018_task1_mediafinal_unet_e120_zoo:largest_cc:a0.00 |
| ISIC / UNet | crc_change_plus_geom | +0.0030 | 0.0009 | 0.158 | 0.006 | 0.0094 | -0.1519 | 0.148 | isic2018_task1_mediafinal_unet_e120_zoo:largest_cc:a0.00 |
| ISIC / UNet | crc_host_uncertainty | +0.0018 | 0.0004 | 0.046 | 0.002 | 0.0039 | -0.1519 | 0.404 | isic2018_task1_mediafinal_unet_e120_zoo:largest_cc:a0.00 |
| ISIC / UNet | crc_quality_risk | +0.0026 | 0.0004 | 0.162 | 0.004 | 0.0040 | -0.0622 | 0.628 | isic2018_task1_mediafinal_unet_e120_zoo:prob_gaussian_lcc:a2.00 |
| ISIC / UNet | random_rate_matched | +0.0029 | 0.0035 | 0.227 | 0.019 | 0.0345 | -0.2824 | 0.032 | isic2018_task1_mediafinal_unet_e120_zoo:largest_cc:a0.00 |
| Kvasir / GraphSeg | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| Kvasir / GraphSeg | best_fixed_cal_utility | +0.0039 | 0.0037 | 0.280 | 0.000 | 0.0256 | -0.0372 | 0.000 | kvasir_seg_mediafinal_graphseg_e120_zoo:prob_gaussian:a2.00 |
| Kvasir / GraphSeg | oracle | +0.0814 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.000 |  |
| Kvasir / GraphSeg | crc_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / GraphSeg | crc_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / GraphSeg | crc_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / GraphSeg | crc_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / GraphSeg | crc_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / GraphSeg | random_rate_matched | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | kvasir_seg_mediafinal_graphseg_e120_zoo:prob_gaussian:a2.00 |
| Kvasir / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| Kvasir / UNet | best_fixed_cal_utility | +0.0021 | 0.0006 | 0.293 | 0.000 | 0.0048 | -0.0134 | 0.000 | kvasir_seg_mediafinal_unet_e120_zoo:remove_small:a250.00 |
| Kvasir / UNet | oracle | +0.0224 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0002 | 0.000 |  |
| Kvasir / UNet | crc_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / UNet | crc_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / UNet | crc_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / UNet | crc_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / UNet | crc_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| Kvasir / UNet | random_rate_matched | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | kvasir_seg_mediafinal_unet_e120_zoo:remove_small:a250.00 |
| MSD Heart MRI / GraphSeg | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| MSD Heart MRI / GraphSeg | best_fixed_cal_utility | +0.0008 | 0.0004 | 0.452 | 0.000 | 0.0023 | -0.0053 | 0.000 | msd_heart_mri_mediafinal_graphseg_mri_e120_zoo:close_only:a3.00 |
| MSD Heart MRI / GraphSeg | oracle | +0.1237 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.032 |  |
| MSD Heart MRI / GraphSeg | crc_changed | +0.0009 | 0.0000 | 0.048 | 0.000 | 0.0000 | -0.0002 | 0.855 | msd_heart_mri_mediafinal_graphseg_mri_e120_zoo:close_only:a11.00 |
| MSD Heart MRI / GraphSeg | crc_geom | -0.0086 | 0.0106 | 0.113 | 0.081 | 0.0940 | -0.2125 | 0.806 | msd_heart_mri_mediafinal_graphseg_mri_e120_zoo:remove_small:a250.00 |
| MSD Heart MRI / GraphSeg | crc_change_plus_geom | -0.0086 | 0.0106 | 0.113 | 0.081 | 0.0940 | -0.2125 | 0.806 | msd_heart_mri_mediafinal_graphseg_mri_e120_zoo:remove_small:a250.00 |
| MSD Heart MRI / GraphSeg | crc_host_uncertainty | -0.0004 | 0.0004 | 0.016 | 0.000 | 0.0037 | -0.0259 | 0.968 | msd_heart_mri_mediafinal_graphseg_mri_e120_zoo:binary_morph:a11.00 |
| MSD Heart MRI / GraphSeg | crc_quality_risk | -0.0185 | 0.0404 | 0.306 | 0.210 | 0.2573 | -0.3884 | 0.355 | msd_heart_mri_mediafinal_graphseg_mri_e120_zoo:binary_morph:a7.00 |
| MSD Heart MRI / GraphSeg | random_rate_matched | +0.0037 | 0.0007 | 0.051 | 0.002 | 0.0061 | -0.0255 | 0.855 | msd_heart_mri_mediafinal_graphseg_mri_e120_zoo:close_only:a11.00 |
| MSD Heart MRI / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| MSD Heart MRI / UNet | best_fixed_cal_utility | -0.0002 | 0.0004 | 0.210 | 0.000 | 0.0028 | -0.0080 | 0.000 | msd_heart_mri_mediafinal_unet_mri_e120_zoo:close_only:a3.00 |
| MSD Heart MRI / UNet | oracle | +0.0484 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.081 |  |
| MSD Heart MRI / UNet | crc_changed | +0.0006 | 0.0002 | 0.145 | 0.000 | 0.0018 | -0.0029 | 0.629 | msd_heart_mri_mediafinal_unet_mri_e120_zoo:open_close:a9.00 |
| MSD Heart MRI / UNet | crc_geom | +0.0004 | 0.0003 | 0.129 | 0.000 | 0.0024 | -0.0037 | 0.677 | msd_heart_mri_mediafinal_unet_mri_e120_zoo:binary_morph:a11.00 |
| MSD Heart MRI / UNet | crc_change_plus_geom | +0.0005 | 0.0003 | 0.129 | 0.000 | 0.0024 | -0.0037 | 0.645 | msd_heart_mri_mediafinal_unet_mri_e120_zoo:binary_morph:a11.00 |
| MSD Heart MRI / UNet | crc_host_uncertainty | -0.0005 | 0.0006 | 0.097 | 0.000 | 0.0051 | -0.0189 | 0.355 | msd_heart_mri_mediafinal_unet_mri_e120_zoo:close_only:a5.00 |
| MSD Heart MRI / UNet | crc_quality_risk | +0.0004 | 0.0000 | 0.032 | 0.000 | 0.0001 | -0.0006 | 0.839 | msd_heart_mri_mediafinal_unet_mri_e120_zoo:binary_morph:a11.00 |
| MSD Heart MRI / UNet | random_rate_matched | -0.0055 | 0.0094 | 0.158 | 0.066 | 0.0823 | -0.2770 | 0.629 | msd_heart_mri_mediafinal_unet_mri_e120_zoo:open_close:a9.00 |
| PH2 / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| PH2 / UNet | best_fixed_cal_utility | +0.0077 | 0.0010 | 0.333 | 0.000 | 0.0069 | -0.0110 | 0.000 | ph2_mediafinal_unet_e120_zoo:largest_cc:a0.00 |
| PH2 / UNet | oracle | +0.0137 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0003 | 0.000 |  |
| PH2 / UNet | crc_changed | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| PH2 / UNet | crc_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| PH2 / UNet | crc_change_plus_geom | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| PH2 / UNet | crc_host_uncertainty | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| PH2 / UNet | crc_quality_risk | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | host |
| PH2 / UNet | random_rate_matched | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 | ph2_mediafinal_unet_e120_zoo:largest_cc:a0.00 |
| Polyp ext. / GraphSeg | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| Polyp ext. / GraphSeg | best_fixed_cal_utility | +0.0086 | 0.0042 | 0.481 | 0.003 | 0.0271 | -0.0633 | 0.000 | polyps_official_mediafinal_graphseg_e120_zoo:close_only:a11.00 |
| Polyp ext. / GraphSeg | oracle | +0.0594 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.018 |  |
| Polyp ext. / GraphSeg | crc_changed | +0.0026 | 0.0004 | 0.148 | 0.000 | 0.0035 | -0.0455 | 0.717 | polyps_official_mediafinal_graphseg_e120_zoo:close_only:a11.00 |
| Polyp ext. / GraphSeg | crc_geom | +0.0015 | 0.0002 | 0.123 | 0.000 | 0.0018 | -0.0250 | 0.779 | polyps_official_mediafinal_graphseg_e120_zoo:close_only:a11.00 |
| Polyp ext. / GraphSeg | crc_change_plus_geom | +0.0013 | 0.0001 | 0.113 | 0.000 | 0.0008 | -0.0196 | 0.810 | polyps_official_mediafinal_graphseg_e120_zoo:close_only:a11.00 |
| Polyp ext. / GraphSeg | crc_host_uncertainty | +0.0001 | 0.0005 | 0.100 | 0.000 | 0.0054 | -0.0350 | 0.857 | polyps_official_mediafinal_graphseg_e120_zoo:close_only:a11.00 |
| Polyp ext. / GraphSeg | crc_quality_risk | +0.0024 | 0.0003 | 0.085 | 0.000 | 0.0027 | -0.0210 | 0.784 | polyps_official_mediafinal_graphseg_e120_zoo:close_only:a11.00 |
| Polyp ext. / GraphSeg | random_rate_matched | +0.0024 | 0.0012 | 0.137 | 0.001 | 0.0119 | -0.0484 | 0.717 | polyps_official_mediafinal_graphseg_e120_zoo:close_only:a11.00 |
| Polyp ext. / UNet | host | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 1.000 |  |
| Polyp ext. / UNet | best_fixed_cal_utility | +0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.000 | polyps_official_mediafinal_unet_e120_zoo:fill_holes:a0.00 |
| Polyp ext. / UNet | oracle | +0.0282 | 0.0000 | 0.000 | 0.000 | 0.0000 | +0.0000 | 0.100 |  |
| Polyp ext. / UNet | crc_changed | +0.0009 | 0.0017 | 0.125 | 0.010 | 0.0171 | -0.1564 | 0.456 | polyps_official_mediafinal_unet_e120_zoo:binary_morph:a9.00 |
| Polyp ext. / UNet | crc_geom | +0.0037 | 0.0018 | 0.120 | 0.010 | 0.0180 | -0.2642 | 0.466 | polyps_official_mediafinal_unet_e120_zoo:binary_morph:a9.00 |
| Polyp ext. / UNet | crc_change_plus_geom | +0.0024 | 0.0021 | 0.125 | 0.013 | 0.0208 | -0.2642 | 0.469 | polyps_official_mediafinal_unet_e120_zoo:binary_morph:a9.00 |
| Polyp ext. / UNet | crc_host_uncertainty | +0.0009 | 0.0015 | 0.088 | 0.008 | 0.0147 | -0.1675 | 0.652 | polyps_official_mediafinal_unet_e120_zoo:binary_morph:a11.00 |
| Polyp ext. / UNet | crc_quality_risk | +0.0004 | 0.0005 | 0.083 | 0.005 | 0.0051 | -0.0534 | 0.474 | polyps_official_mediafinal_unet_e120_zoo:remove_small:a250.00 |
| Polyp ext. / UNet | random_rate_matched | -0.0037 | 0.0145 | 0.147 | 0.050 | 0.1440 | -0.5795 | 0.466 | polyps_official_mediafinal_unet_e120_zoo:binary_morph:a9.00 |
