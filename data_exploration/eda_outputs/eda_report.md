# Sport-survey exploratory data analysis

All label comparisons and plots below use the **training split only**. Respondent IDs are split as groups to avoid putting the same person's answers in more than one split.

## 1. Dataset and split integrity
- Raw rows: 870; retained rows: 861; all-answer-blank rows removed: 7.
- Unique respondents: 287; training respondents: 201; training rows: 603.
- Most common label sequences per respondent: {('Soccer', 'Baseball', 'Basketball'): 287}.
- The ID is an identifier, not a predictor; exclude it from model inputs.

Split summary:
       respondents  rows
split                   
dev            201   603
test            86   258

## 2. Label balance
            rows  percent
label                    
Soccer       201    33.33
Baseball     201    33.33
Basketball   201    33.33
Interpretation: the class balance is approximately even, so accuracy is a reasonable baseline metric.

## 3. Numeric feature distributions and label relationships
- **Ease rating (0–10):** n=603/603 non-missing; median=6.00; IQR=4.00–7.00; range=0.00–10.00; skewness=-0.08.
  By sport (mean / median): Soccer: 6.36 / 7.00; Baseball: 4.11 / 4.00; Basketball: 6.15 / 6.00
  Largest difference in sport means: 2.25. This is descriptive and does not establish causation.
- **Willingness to pay ($):** n=563/603 non-missing; median=100.00; IQR=50.00–200.00; range=0.00–10000.00; skewness=9.78.
  By sport (mean / median): Soccer: 364.30 / 200.00; Baseball: 128.83 / 50.00; Basketball: 255.27 / 150.00
  Largest difference in sport means: 235.48. This is descriptive and does not establish causation.
- **log(1 + willingness to pay):** n=563/603 non-missing; median=4.62; IQR=3.93–5.30; range=0.00–9.21; skewness=-1.13.
  By sport (mean / median): Soccer: 4.85 / 5.30; Baseball: 3.80 / 3.93; Basketball: 4.70 / 5.02
  Largest difference in sport means: 1.05. This is descriptive and does not establish causation.
- Price is typically right-skewed: raw-price skewness=9.78, log-price skewness=-1.13. The log transform reduces the absolute skewness.
- On the log-IQR rule, 10 values exceed approximately $1,572. Inspect extreme values; do not automatically delete genuine high bids.

## 4. Ordinal (Likert) responses
- **Success in this sport depends heavily on strategy and planning.** Mean ordinal score: Soccer 3.94; Baseball 3.57; Basketball 3.88.
  Within-sport response percentages (columns 1=strongly disagree to 5=strongly agree):
Success in this sport depends heavily on strategy and planning. [ordinal]    1     2     3     4     5
label                                                                                                 
Soccer                                                                     1.0   7.0  11.4  58.2  22.4
Baseball                                                                   1.0  16.9  27.4  33.3  21.4
Basketball                                                                 0.5   5.0  19.5  55.5  19.5
- **Players must be comfortable with a high risk of injury.** Mean ordinal score: Soccer 3.75; Baseball 2.87; Basketball 3.76.
  Within-sport response percentages (columns 1=strongly disagree to 5=strongly agree):
Players must be comfortable with a high risk of injury. [ordinal]    1     2     3     4     5
label                                                                                         
Soccer                                                             1.0  12.4  19.4  45.3  21.9
Baseball                                                           5.5  34.3  34.3  19.4   6.5
Basketball                                                         1.5  10.4  21.9  42.8  23.4
- **Individual brilliance can outweigh teamwork in this sport.** Mean ordinal score: Soccer 2.89; Baseball 3.75; Basketball 3.62.
  Within-sport response percentages (columns 1=strongly disagree to 5=strongly agree):
Individual brilliance can outweigh teamwork in this sport. [ordinal]    1     2     3     4     5
label                                                                                            
Soccer                                                                9.0  32.3  27.4  23.4   8.0
Baseball                                                              2.0  13.4  17.9  41.3  25.4
Basketball                                                            2.5  14.0  21.5  42.5  19.5

## 5. Multi-select answer choices
Options with the largest between-sport differences (percentage-point spread):
                                                                                   question                        option  Soccer_pct  Baseball_pct  Basketball_pct  spread_percentage_points
                                                           Where might you play this sport?               Indoor facility        66.2          23.4            96.0                      72.6
                                 What types of music do you most associate with this sport?                Hip-hop or rap        41.8          25.4            84.6                      59.2
                                                           Where might you play this sport?             At home or nearby        47.3          12.4            67.7                      55.2
During which seasons would you be most likely to play this sport recreationally in Toronto?                        Winter         6.5           5.0            59.2                      54.2
                                                           Where might you play this sport? Community recreation facility        77.6          48.8            89.6                      40.8
                                 What types of music do you most associate with this sport?                         Latin        41.3           4.5             0.5                      40.8
                                 What types of music do you most associate with this sport?                       Country         4.0          33.8             3.0                      30.8
                                                           Where might you play this sport?       Private sports facility        69.2          56.2            84.1                      27.9
                                 What types of music do you most associate with this sport?            No particular type        17.4          31.3             9.5                      21.9
                                 What types of music do you most associate with this sport?           Electronic or dance        36.3          14.4            20.4                      21.9
During which seasons would you be most likely to play this sport recreationally in Toronto?                          Fall        69.2          74.1            88.1                      18.9
                                                           Where might you play this sport? School or university facility        89.6          76.6            94.5                      17.9
During which seasons would you be most likely to play this sport recreationally in Toronto?                        Spring        69.7          74.6            84.1                      14.4
                                                           Where might you play this sport?                        Others        15.9           4.0            11.4                      11.9
                                 What types of music do you most associate with this sport?                           Pop        39.8          36.3            46.8                      10.4

## 6. Missing answers and response patterns
                                                                                                                                                                                                          question  overall_missing_pct  Soccer_missing_pct  Baseball_missing_pct  Basketball_missing_pct  max_min_gap_pp
                                                                                                                                                                      Which song do you associate with this sport?                 10.1                 4.0                  14.9                    11.4            10.9
                                                                                                            What foods and drinks do you associate with watching this sport in the place or culture you know best?                  1.8                 0.0                   2.0                     3.5             3.5
                                                                                                                                                                     What colour do you associate with this sport?                  1.0                 1.0                   1.5                     0.5             1.0
                                                                                                                                                                        How do you feel when you watch this sport?                  0.7                 0.0                   1.5                     0.5             1.5
                                                                                                               How much would you be willing to pay for a decent seat at a professional competition of this sport?                  0.5                 0.5                   0.5                     0.5             0.0
                                                                                                                                                        Individual brilliance can outweigh teamwork in this sport.                  0.2                 0.0                   0.0                     0.5             0.5
                                                                                                                                                   Success in this sport depends heavily on strategy and planning.                  0.2                 0.0                   0.0                     0.5             0.5
How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy.                  0.0                 0.0                   0.0                     0.0             0.0
                                                                                                                                                           Players must be comfortable with a high risk of injury.                  0.0                 0.0                   0.0                     0.0             0.0
                                                                                                                                                        What types of music do you most associate with this sport?                  0.0                 0.0                   0.0                     0.0             0.0
                                                                                                                                                                                  Where might you play this sport?                  0.0                 0.0                   0.0                     0.0             0.0
                                                                                                                       During which seasons would you be most likely to play this sport recreationally in Toronto?                  0.0                 0.0                   0.0                     0.0             0.0
Interpretation: label-dependent missingness can itself become predictive, but may also reflect survey-taking behaviour. Compare a model with and without missingness indicators.

## 7. Free-response answers: vocabulary and label associations
- **What foods and drinks do you associate with watching this sport in the place or culture you know best?** Answer rate=98.0%; distinct normalized answers=439; mean words=4.72; median words=3.0; vocabulary size=399.
- **What colour do you associate with this sport?** Answer rate=99.0%; distinct normalized answers=138; mean words=2.04; median words=1.0; vocabulary size=150.
- **Which song do you associate with this sport?** Answer rate=89.6%; distinct normalized answers=340; mean words=4.49; median words=4.0; vocabulary size=553.
- **How do you feel when you watch this sport?** Answer rate=99.3%; distinct normalized answers=338; mean words=4.83; median words=1.0; vocabulary size=640.

Highest-MI tokens for **What foods and drinks do you associate with watching this sport in the place or culture you know best?** (exploratory; not causal):
  token  n_rows  Soccer_pct  Baseball_pct  Basketball_pct  mi_bits
    hot     136      11.940        44.279          11.443    0.093
   dogs     101       8.458        34.826           6.965    0.080
   beer     195      47.264        30.846          18.905    0.045
  wings      55       7.960         1.493          17.910    0.044
peanuts      26       0.995        11.443           0.498    0.043
  pizza      77      16.418         3.980          17.910    0.030
  chips      75      20.896         6.468           9.950    0.024
   fish       8       3.980         0.000           0.000    0.021
   coca      14       5.970         0.498           0.498    0.020
chicken      69      11.940         4.975          17.413    0.020
popcorn      91       7.463        18.408          19.403    0.018
hotdogs      39       5.970        10.448           2.985    0.011

Highest-MI tokens for **What colour do you associate with this sport?** (exploratory; not causal):
 token  n_rows  Soccer_pct  Baseball_pct  Basketball_pct  mi_bits
orange     119       0.498         1.493          57.214    0.336
 green     118      55.224         3.483           0.000    0.310
  blue     153      19.900        53.731           2.488    0.189
 white      99      19.403        28.358           1.493    0.084
 black      46      12.438         0.995           9.453    0.031
 brown      40       0.995         9.453           9.453    0.024
purple       7       0.000         0.000           3.483    0.019
   and      50      13.930         4.478           6.468    0.015
   red     127      22.388        13.433          27.363    0.015
yellow      27       7.463         1.493           4.478    0.011
  gray       3       0.000         1.493           0.000    0.008
 which       3       1.493         0.000           0.000    0.008

Highest-MI tokens for **Which song do you associate with this sport?** (exploratory; not causal):
  token  n_rows  Soccer_pct  Baseball_pct  Basketball_pct  mi_bits
   waka     102      50.746         0.000           0.000    0.322
   take      61       0.000        30.348           0.000    0.177
    out      57       0.000        28.358           0.000    0.165
     to      64       0.000        30.348           1.493    0.156
shakira      53      26.368         0.000           0.000    0.152
   ball      57       0.000        27.861           0.498    0.152
     me      66       1.493        30.846           0.498    0.149
   game      56       0.000        27.363           0.498    0.149
   time      30      14.925         0.000           0.000    0.083
 africa      29      14.428         0.000           0.000    0.080
    for      31      14.925         0.000           0.498    0.075
    the     102       5.970        33.831          10.945    0.073

Highest-MI tokens for **How do you feel when you watch this sport?** (exploratory; not causal):
    token  n_rows  Soccer_pct  Baseball_pct  Basketball_pct  mi_bits
  excited     180      42.289         8.458          38.806    0.091
    bored      76       7.463        24.876           5.473    0.047
  relaxed      18       0.000         8.458           0.498    0.039
     fast      14       0.000         0.000           6.965    0.038
    paced       9       0.000         0.000           4.478    0.024
 confused      17       0.498         6.965           0.995    0.022
   boring      16       2.488         5.473           0.000    0.019
     calm       6       0.000         2.985           0.000    0.016
energetic      22       3.980         0.498           6.468    0.015
energized      12       1.990         0.000           3.980    0.013
     pace       5       0.000         0.000           2.488    0.013
       at       9       0.995         3.483           0.000    0.012
Free-response interpretation: exact-answer counts reveal recurring associations; token prevalence and mutual information reveal terms whose presence differs across labels. Rare terms can appear highly distinctive by chance, so inspect support counts and validate any selected text features using cross-validation grouped by respondent.

## 8. Relationships among numeric and ordinal features
Spearman correlation matrix (monotonic associations; pooled across sports):
                                                                           ease_numeric  Success in this sport depends heavily on strategy and planning. [ordinal]  Players must be comfortable with a high risk of injury. [ordinal]  Individual brilliance can outweigh teamwork in this sport. [ordinal]  log_price
ease_numeric                                                                       1.00                                                                       0.03                                                               0.11                                                                 -0.08       0.17
Success in this sport depends heavily on strategy and planning. [ordinal]          0.03                                                                       1.00                                                               0.32                                                                 -0.21       0.23
Players must be comfortable with a high risk of injury. [ordinal]                  0.11                                                                       0.32                                                               1.00                                                                 -0.09       0.24
Individual brilliance can outweigh teamwork in this sport. [ordinal]              -0.08                                                                      -0.21                                                              -0.09                                                                  1.00      -0.05
log_price                                                                          0.17                                                                       0.23                                                               0.24                                                                 -0.05       1.00
Strongest absolute pooled Spearman association: Success in this sport depends heavily on strategy and planning. [ordinal] vs Players must be comfortable with a high risk of injury. [ordinal], rho=0.32. Pooled correlations can be driven by label differences; inspect within-sport correlations before claiming respondent-level relationships.

## 9. Exploratory feature-label association ranking

## 10. Implications for model development
1. Start with a structured-only baseline (one-hot categorical options + ordinal/numeric features) and multinomial logistic regression.
2. Compare against TF-IDF on free-response columns, then a combined structured + text model.
3. Fit imputers, scalers, encoders, and text vocabularies on training folds only. Use respondent-grouped cross-validation.
4. Use macro-F1, accuracy, per-class precision/recall, and a confusion matrix. Keep the test set untouched until final model selection.
5. Remove respondent ID from predictors. Consider a sensitivity analysis with and without missingness indicators.
6. Treat all rankings here as exploratory. They describe association, not causation, and small samples/rare words can create unstable apparent signals.

## Output inventory
- Tables: `data_exploration/eda_outputs/`
- Figures: `data_exploration/figures/`
- Main plots: label balance; numeric boxplots/distributions; Likert distributions; multi-select option rates; missingness; free-text response lengths and token rates; correlation heatmap; mutual-information ranking.
