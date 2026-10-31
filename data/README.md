# Data

This project uses the **Next Day Wildfire Spread** dataset: 64 x 64 km patches of remote-sensing and weather data at 1 km resolution, with the fire mask for the previous day and the next day.

The data is **not stored in this repository**.

## Download

The original notebook read it from the Kaggle dataset `next-day-wildfire-spread`. Download it from Kaggle and place the TFRecord files in this folder:

```
data/
├── next_day_wildfire_spread_train_00.tfrecord ... 
├── next_day_wildfire_spread_eval_00.tfrecord  ...
└── next_day_wildfire_spread_test_00.tfrecord  ...
```

File names must match the globs `next_day_wildfire_spread_{train,eval,test}*`. Check the dataset page for its license and terms before reuse.

## Channels

| # | Feature | Meaning |
|---|---|---|
| 1 | `elevation` | Digital elevation model (DEM) |
| 2 | `th` | Wind direction |
| 3 | `vs` | Wind velocity |
| 4 | `tmmn` | Minimum temperature |
| 5 | `tmmx` | Maximum temperature |
| 6 | `sph` | Specific humidity |
| 7 | `pr` | Precipitation |
| 8 | `pdsi` | Drought index |
| 9 | `NDVI` | Vegetation index |
| 10 | `population` | Population density |
| 11 | `erc` | Energy release component |
| 12 | `PrevFireMask` | Fire mask of the previous day |

Target: `FireMask` (next day). Unlabeled pixels (-1) are treated as "no fire".

## Citation

F. Huot, R. L. Hu, N. Goyal, T. Sankar, M. Ihme, Y.-F. Chen. *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data.* IEEE Transactions on Geoscience and Remote Sensing, 2022.
