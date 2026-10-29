# Smart Traffic Congestion Analysis and Prediction

EDA project for BCSE331L (Exploratory Data Analysis).

We used the UCI Metro Interstate Traffic Volume dataset, which has hourly traffic counts for one westbound stretch of I-94 (Minneapolis–St Paul) from October 2012 to September 2018, along with weather and holiday information. The aim was to understand when and why traffic is heavy, find recurring traffic patterns, and then predict the next hour's traffic volume.

One thing to keep in mind: the dataset only records how many vehicles passed, not their speed or delay. So whenever we talk about "congestion" or Low/Medium/High traffic, we mean traffic volume, not measured congestion.

Dataset: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put `Metro_Interstate_Traffic_Volume.csv` in `data/raw/`, then run the phases in order (each one uses the output of the earlier ones):

```
.venv/bin/python scripts/phase1_data_transformation.py
.venv/bin/python scripts/phase2_correlation_timeseries.py
.venv/bin/python scripts/phase3_summarization_visualization.py
.venv/bin/python scripts/phase4_clustering_outliers.py
.venv/bin/python scripts/phase5_dimensionality_reduction.py
.venv/bin/python scripts/phase6_regression_model.py
```

Results are printed to the terminal and the plots are saved in `outputs/phase1` to `outputs/phase6`. The cleaned data goes to `data/processed/`.

## Phase 1 – Data Transformation (Module 2)

The raw file has 48,204 rows, but it isn't one row per hour. 17 rows were exact duplicates, and another 7,612 rows repeated an hour that was already there, because the same hour is listed once for every weather condition (e.g. "Rain" and "Mist") with the same traffic count. We kept one row per hour, which left 40,575 hours.

The holiday column only had the holiday name on the midnight row, so we copied it to the rest of that day.

There were no actual NaN values, but 10 rows had a temperature of 0 Kelvin and one row had 9,831 mm of rain in an hour, which are clearly sensor errors. We treated them as missing and compared two options: dropping those rows or filling them with the median. Dropping loses 11 hours of valid traffic data, and the median barely changed any statistic, so we used median imputation.

About 23% of hours are missing from the timeline, mostly because of a long sensor outage from August 2014 to June 2015. We did not try to fill these in.

For binning, traffic volume was split into Low (< 2000 vehicles/hour), Medium (2000–4499) and High (≥ 4500). These are close to the tertiles of the data, so each level has roughly a third of the hours. We also binned temperature into three bands and the hour into five parts of the day (night, AM peak, midday, PM peak, evening).

## Phase 2 – Correlation and Time Series (Module 3)

We looked at the distributions of each variable, the correlations with traffic volume, then set a datetime index and resampled to daily and monthly averages.

- Weather barely correlates with traffic volume (temperature r = 0.14, rain and snow close to 0).
- Hour of day has the biggest effect, but its correlation is only 0.36 because the relationship is a curve: low at night, high during the day.
- Weekdays have two rush hours, at 7:00 (about 6,200 veh/h) and 16:00 (about 6,300 veh/h). Weekends have no morning rush and just one broad peak around 13:00.
- Average weekday traffic is 37% higher than on weekends. Holidays on average look like weekends.
- Seasonal variation is small. December and January are the lowest months, but the difference between the highest and lowest month is only about 13%.
- Traffic in one hour has a correlation of 0.91 with the previous hour, which is why we used past traffic as a feature in Phase 6.

## Phase 3 – Summarization and Visualization (Module 4)

We computed the mean, median, standard deviation, quartiles and IQR outliers for the numeric columns. Traffic volume has no outliers using the usual 1.5 × IQR rule because its range is very wide (night and day are mixed). But if you compare each hour only with the same hour and day type, about 1,480 hours (3.6%) stand out as unusual.

Contingency tables with chi-square tests showed:

- 46% of weekday hours are High, compared to 12% on weekends and 15% on holidays.
- Day type is more strongly associated with traffic level (Cramér's V = 0.24) than weather is (V = 0.11).
- Time of day matters even more: 65% of AM peak hours are High, while 89% of night hours are Low.

We also made a scatter plot (traffic vs hour), a dot chart (median traffic by day and time of day) and a stacked bar plot (traffic level mix by time of day).

## Phase 4 – Clustering and Outlier Detection (Module 5)

Here each day was one observation, described by its average traffic in the five parts of the day. We fit a Gaussian Mixture Model with the EM algorithm. BIC dropped a lot at both K = 2 and K = 4, but at K = 4 the extra cluster only split normal weekdays into early-week and late-week groups, so we went with K = 3.

The three clusters:

- **Normal weekday** (955 days) – strong morning and evening peaks.
- **Weekend / holiday** (425 days) – no morning rush, one midday hump.
- **Reduced weekday** (158 days) – in between, with a much weaker morning rush. This group shows up more in December, January and July, and 17% of its hours had snow compared with 5% overall. Just looking at weekday/weekend labels wouldn't have shown this group.

For outliers we took the 1% of days with the lowest likelihood under the model (16 days). These include heavy snow days (for example 2018-04-14, where 83% of the hours had snow) and a few days with almost no traffic in July 2016, which could be a road closure or a sensor problem. Some outlier days have no obvious explanation in the data.

## Phase 5 – Dimensionality Reduction (Module 6)

We ran PCA on 8 standardized features: traffic volume, temperature, rain, snow, clouds, hour (as sine and cosine) and a weekend flag.

- PC1 explains 24% of the variance and is basically the daily traffic cycle (traffic volume together with time of day).
- PC2 is temperature vs cloud cover, PC3 is rain and clouds, and PC4 is snow and weekend.
- It takes 6 of the 8 components to reach 80% of the variance, so the features aren't very redundant. Traffic volume loads almost only on PC1, which matches what we saw in Phase 2: weather doesn't share much with traffic.
- In the PC1–PC2 plot, Low, Medium and High hours line up from left to right along PC1.

We then did an SVD of the same standardized matrix to check the link with PCA. The squared singular values divided by (n − 1) are exactly the PCA eigenvalues, the rows of Vᵀ are the PCA components and U·S gives the PCA scores. The matrix has full rank and a condition number of 3.45, so no feature is nearly a copy of another.

## Phase 6 – Regression and Evaluation (Module 7)

We used linear regression to predict the traffic volume of the next hour. The features only use information that would be available at the time of prediction: traffic now, one hour ago, the same hour yesterday and last week, the current weather, and the hour/weekend/holiday of the next hour. The data was split by time: training on Oct 2012 – Sep 2017 and testing on the last 12 months (Oct 2017 – Sep 2018), so the model never sees future data.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline: next hour = this hour | 591 | 819 | 0.827 |
| Baseline: same hour last week | 344 | 660 | 0.888 |
| Linear regression (calendar + weather) | 592 | 793 | 0.838 |
| Linear regression (+ traffic history) | 242 | 350 | 0.968 |

The full model beats both baselines and has about 59% lower MAE than just repeating the current hour. Train and test scores were almost the same, so it isn't overfitting. Without past traffic, R² only reaches 0.84, so recent traffic is the most useful input.

The biggest errors happen in the early morning (5–8 am) when traffic rises very quickly, and on holidays and snow days. The single worst day was the 2018-04-14 snowstorm, which was also an outlier in Phase 4.

## Conclusions

1. Traffic on this road is driven mostly by time: hour of day and whether it is a weekday. Weekdays have two rush hours and weekends have one midday peak.
2. Weather has a weak effect on traffic volume overall. Some of the apparent weather effects are really time-of-day effects (fog and mist happen mostly at night).
3. Clustering found a hidden "reduced weekday" pattern linked to winter, the holiday season and snow, which simple day labels don't show.
4. Unusual days (snowstorms, near-zero traffic days) can be picked up automatically as outliers from the clustering model.
5. A simple linear regression using recent traffic and the calendar predicts next-hour volume well (R² = 0.968), much better than naive baselines.

## Limitations

- The data covers a single road segment, so the results may not apply to a whole city.
- Only traffic volume is measured, so the traffic levels are a stand-in for congestion, not a real measure of it.
- Almost a year of data (2014–15) is missing.
- The holiday flag treats every holiday the same, even though some holidays still have normal commuter traffic.

## Project structure

```
data/raw/          original dataset
data/processed/    cleaned data and daily clusters
scripts/           phase1 ... phase6 scripts
outputs/           plots for each phase
requirements.txt
```
