# Entry Filter Rule V1

Research-derived threshold rules for Direction Champion entries.
**Not prediction. No ML. Scan-time features only.**

Signals analyzed: long=897 short=897
Winner/Loser split: top/bottom 20%

## LONG Entry Filter Rule V1

```
IF
  1h_current_return_pct >= 17.579276  AND
  1h_current_body_pct >= 13.83363  AND
  1h_current_range_pct >= 26.929756  AND
  2h_current_body_pct >= 22.121308
THEN
  PASS
ELSE
  REJECT
```

- Combined pass n=11 | precision=1.0 | recall=0.0615 | f1=0.1158
- Avg return 2h (pass)=28.8611% | 4h=28.7559% | win_rate=100.0%

## SHORT Entry Filter Rule V1

```
IF
  1h_current_return_pct <= -11.76382  AND
  1h_current_body_pct >= 10.89637  AND
  2h_current_return_pct <= -20.089984  AND
  1h_current_range_pct >= 17.41933
THEN
  PASS
ELSE
  REJECT
```

- Combined pass n=11 | precision=1.0 | recall=0.0615 | f1=0.1158
- Avg return 2h (pass)=23.7559% | 4h=25.9786% | win_rate=100.0%

## Per-feature conditions

### Long
- `1h_current_return_pct` >= **17.579276** | f1=0.2157 lift=4.4098 use=True | avg2h_pass=25.5408%
- `1h_current_body_pct` >= **13.83363** | f1=0.3056 lift=4.4694 use=True | avg2h_pass=22.4044%
- `1h_current_range_pct` >= **26.929756** | f1=0.1436 lift=4.3848 use=True | avg2h_pass=23.4209%
- `2h_current_body_pct` >= **22.121308** | f1=0.2464 lift=4.0716 use=True | avg2h_pass=22.0346%

### Short
- `1h_current_return_pct` <= **-11.76382** | f1=0.3005 lift=4.7164 use=True | avg2h_pass=20.1338%
- `1h_current_body_pct` >= **10.89637** | f1=0.3482 lift=4.343 use=True | avg2h_pass=14.663%
- `2h_current_return_pct` <= **-20.089984** | f1=0.1538 lift=4.698 use=True | avg2h_pass=19.3033%
- `1h_current_range_pct` >= **17.41933** | f1=0.366 lift=3.8479 use=True | avg2h_pass=12.653%