# Word recognizer - offline results
Offline held-out clip evaluation, not measured live/webcam accuracy.

Written by check_results.py on 20261122-145305. Read-only: it re-ran nothing.

## Headline

| | |
|---|---|
| test top-1 (official signer-disjoint) | 68.3% |
| test top-5 | 87.6% |
| test macro over words | 81.3% |
| test top-1 with REST/OTHER removed from the argmax | 81.4% |
| words-only top-1: train vs val (memorisation gap) | 96.9% vs 81.4% |
| best val top-1 | 62.5% |
| test size | 1319 samples / 1022 videos / 11 signers |
| train size | 4572 samples / 91 signers |
| epochs | 56 (best val 62.5%) |
| features | 200 dims, blocks all, 64 frames |
| classes | 58 |
| label-set identity | 490d04657dd08d4fb052 |
| evaluation identity | 9f198832eda8c3fefb3c |
| parameters | 385178 |
| seed / seconds | 20261118 / 809.4 (14.2 s per epoch) |

Every test number above is from signers the model never saw. A random split would read far higher and would not be reported here.

## Held-out groups (never pooled)

| group | videos | top-1 | words-only top-1 |
|---|---|---|---|
| ASL Citizen official test signers (offline) | 1022 | 68.3% | 81.4% |
| PopSign val+test signers (second, independent style) | 840 | 83.7% | 83.7% |

Training saw 2520 PopSign clips on top of the ASL Citizen train set. Each row is the signer-disjoint split of its own corpus; a number that pooled them would be meaningless and is not reported. Neither row is a measurement of live webcam accuracy.

## Signer-disjointness check

- no problem found: no (source, signer) pair appears in train and in a held-out split

## Retention and expansion (full-classifier argmax)

Core20 is active batch 1; new_words is the active later batches. Expanded-total accuracy is not an improvement over an old smaller-vocabulary baseline.
Weak-six membership: six weakest core20 words on this run's official val split; fixed from VAL for every final split.

| split | group | samples | words scored | top-1 | macro | correct-commit recall | labels |
|---|---|---|---|---|---|---|---|
| val | core20 | 84 | 20 | 83.3% | 81.5% | 59.5% | eat, family, friend, good, hello, help, name, sad, school, drink, thanks, love, bye, sorry, yes, no, happy, please, bad, home |
| val | weak6 | 22 | 6 | 54.5% | 54.2% | 18.2% | happy, thanks, please, no, bad, good |
| val | new_words | 136 | 35 | 79.4% | 80.3% | 58.1% | me, my, you, your, what, where, who, want, need, like, go, come, stop, more, water, understand, now, today, work, wait, finish, sleep, bathroom, hungry, thirsty, when, why, how, apple, bread, chicken, tomorrow, sick, hurt, doctor, medicine |
| test | core20 | 269 | 20 | 83.6% | 83.0% | 49.1% | eat, family, friend, good, hello, help, name, sad, school, drink, thanks, love, bye, sorry, yes, no, happy, please, bad, home |
| test | weak6 | 73 | 6 | 60.3% | 60.8% | 16.4% | happy, thanks, please, no, bad, good |
| test | new_words | 462 | 35 | 79.9% | 80.3% | 58.0% | me, my, you, your, what, where, who, want, need, like, go, come, stop, more, water, understand, now, today, work, wait, finish, sleep, bathroom, hungry, thirsty, when, why, how, apple, bread, chicken, tomorrow, sick, hurt, doctor, medicine |
| ps_test | core20 | 320 | 16 | 83.1% | 83.1% | 52.2% | eat, family, friend, good, hello, help, name, sad, school, drink, thanks, love, bye, sorry, yes, no, happy, please, bad, home |
| ps_test | weak6 | 120 | 6 | 76.7% | 76.7% | 40.8% | happy, thanks, please, no, bad, good |
| ps_test | new_words | 520 | 26 | 84.0% | 84.0% | 60.8% | me, my, you, your, what, where, who, want, need, like, go, come, stop, more, water, understand, now, today, work, wait, finish, sleep, bathroom, hungry, thirsty, when, why, how, apple, bread, chicken, tomorrow, sick, hurt, doctor, medicine |

## Fixed commit policy (offline clips)

Policy is saved with this run; test is not re-thresholded by this report. Missing historical policy metrics are not reconstructed from rounded CSVs.
Saved policy: `{"confidence_min": 0.85, "hedge_pairs": {"good|thanks": "good/thanks?"}, "margin_min": 0.15, "pair_margins": {}, "scope": "offline clip decisions; not measured live accuracy", "version": "predictor.decide/commitable-v1"}`

| split | commits | precision | word commit share | correct-commit recall | hedge refusals | REST slip | OTHER slip |
|---|---|---|---|---|---|---|---|
| val | 166 | 77.7% | 63.6% | 58.6% | 1 | 2.7% | 14.7% |
| test | 464 | 86.2% | 58.1% | 54.7% | 3 | 2.3% | 11.1% |
| ps_test | 495 | 97.6% | 58.9% | 57.5% | 9 | n/a | n/a |

## Background classes (raw argmax, before commit policy)

| | |
|---|---|
| REST predicted as a word | 12.3% |
| OTHER predicted as a word | 83.7% |
| a real word predicted as REST/OTHER | 1.4% |

## Weakest words (test description only, not selection)

- thanks 25.0% (12 samples, 11 signers)
- hungry 33.3% (12 samples, 11 signers)
- thirsty 41.7% (12 samples, 11 signers)
- please 46.2% (13 samples, 11 signers)
- doctor 46.2% (13 samples, 11 signers)
- happy 50.0% (12 samples, 11 signers)

## Most confused pairs

- what / want: 7 swapped of 49 (14.3%)
- now / today: 6 swapped of 23 (26.1%)
- please / my: 6 swapped of 26 (23.1%)
- want / wait: 5 swapped of 37 (13.5%)
- sorry / me: 4 swapped of 25 (16.0%)
- happy / my: 4 swapped of 25 (16.0%)
- help / doctor: 4 swapped of 26 (15.4%)
- good / thanks: 3 swapped of 24 (12.5%)

## Hand detection per word (worst window rate first)

| word | signing window | whole clip | videos |
|---|---|---|---|
| me | 0.464 | 0.324 | 31 |
| sorry | 0.663 | 0.503 | 30 |
| stop | 0.714 | 0.375 | 131 |
| come | 0.725 | 0.297 | 30 |
| like | 0.733 | 0.397 | 128 |
| hungry | 0.751 | 0.391 | 129 |
| bad | 0.760 | 0.362 | 130 |
| hurt | 0.765 | 0.400 | 130 |
| thirsty | 0.778 | 0.427 | 127 |
| why | 0.783 | 0.411 | 132 |
| good | 0.788 | 0.399 | 131 |
| what | 0.790 | 0.423 | 163 |
| chicken | 0.800 | 0.470 | 100 |
| love | 0.804 | 0.445 | 130 |
| sick | 0.804 | 0.402 | 131 |
| when | 0.805 | 0.422 | 131 |
| hello | 0.806 | 0.387 | 132 |
| my | 0.811 | 0.416 | 30 |
| sleep | 0.811 | 0.385 | 130 |
| more | 0.815 | 0.441 | 131 |
| tomorrow | 0.822 | 0.383 | 128 |
| go | 0.823 | 0.389 | 131 |
| doctor | 0.826 | 0.457 | 119 |
| train | 0.830 | 0.485 | 32 |

The window column is the one to read: it is measured between sign_start and sign_end, where the sign actually happens. The clip column is what the extractor reports for the whole sentence clip, idle head and tail included. A word can only be missed because of detection if the window column is low - and per reports/predictions.csv it usually is not.

## Extraction errors

- none recorded

## Seed sweep: baseline vs new

**none-20261122-1303** - 3 seed(s), reg=none, feature v3, promoted seed 20261118

label_set_id: 490d04657dd08d4fb052
evaluation_id: 9f198832eda8c3fefb3c

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.809 | 0.030 | 0.777..0.836 |
| val_top1 | 0.599 | 0.030 | 0.565..0.625 |
| val_typed_share | 0.564 | 0.114 | 0.414..0.641 |
| val_typed_precision | 0.768 | 0.008 | 0.761..0.777 |
| val_other_slip | 0.131 | 0.037 | 0.087..0.160 |
| val_rest_slip | 0.033 | 0.007 | 0.027..0.040 |
| test_words_top1 | 0.793 | 0.038 | 0.746..0.821 |
| test_words_top5 | 0.959 | 0.003 | 0.955..0.962 |
| test_top1 | 0.646 | 0.045 | 0.594..0.683 |
| test_macro_words | 0.792 | 0.034 | 0.747..0.815 |
| test_typed_share | 0.534 | 0.095 | 0.414..0.605 |
| test_typed_precision | 0.852 | 0.036 | 0.811..0.884 |
| test_typed_precision_words | 0.932 | 0.020 | 0.908..0.948 |
| test_other_slip | 0.100 | 0.012 | 0.087..0.111 |
| test_rest_slip | 0.023 | 0.013 | 0.010..0.037 |
| test_other_as_word | 0.924 | 0.080 | 0.837..0.997 |
| ps_test_words_top1 | 0.787 | 0.057 | 0.724..0.837 |
| ps_test_top1 | 0.787 | 0.057 | 0.724..0.837 |
| ps_test_typed_share | 0.467 | 0.158 | 0.273..0.589 |
| ps_test_typed_precision_words | 0.965 | 0.011 | 0.954..0.976 |
| ps_test_core20_correct_commit_recall | 0.415 | 0.138 | 0.247..0.522 |
| ps_test_core20_macro_top1 | 0.795 | 0.034 | 0.762..0.831 |
| ps_test_core20_n_samples | 320.000 | 0.000 | 320.000..320.000 |
| ps_test_core20_top1 | 0.795 | 0.034 | 0.762..0.831 |
| ps_test_new_words_correct_commit_recall | 0.474 | 0.167 | 0.273..0.608 |
| ps_test_new_words_macro_top1 | 0.783 | 0.070 | 0.700..0.840 |
| ps_test_new_words_n_samples | 520.000 | 0.000 | 520.000..520.000 |
| ps_test_new_words_top1 | 0.783 | 0.070 | 0.700..0.840 |
| ps_test_weak6_correct_commit_recall | 0.312 | 0.121 | 0.167..0.408 |
| ps_test_weak6_macro_top1 | 0.728 | 0.046 | 0.675..0.767 |
| ps_test_weak6_n_samples | 106.667 | 20.000 | 80.000..120.000 |
| ps_test_weak6_top1 | 0.728 | 0.046 | 0.675..0.767 |
| test_core20_correct_commit_recall | 0.492 | 0.117 | 0.376..0.610 |
| test_core20_macro_top1 | 0.801 | 0.048 | 0.739..0.835 |
| test_core20_n_samples | 269.000 | 0.000 | 269.000..269.000 |
| test_core20_top1 | 0.809 | 0.048 | 0.747..0.844 |
| test_new_words_correct_commit_recall | 0.503 | 0.102 | 0.377..0.580 |
| test_new_words_macro_top1 | 0.786 | 0.026 | 0.751..0.803 |
| test_new_words_n_samples | 462.000 | 0.000 | 462.000..462.000 |
| test_new_words_top1 | 0.781 | 0.027 | 0.745..0.799 |
| test_weak6_correct_commit_recall | 0.313 | 0.226 | 0.162..0.613 |
| test_weak6_macro_top1 | 0.628 | 0.102 | 0.537..0.740 |
| test_weak6_n_samples | 74.000 | 1.000 | 73.000..75.000 |
| test_weak6_top1 | 0.630 | 0.103 | 0.540..0.747 |
| val_core20_correct_commit_recall | 0.512 | 0.125 | 0.345..0.595 |
| val_core20_macro_top1 | 0.795 | 0.048 | 0.737..0.833 |
| val_core20_n_samples | 84.000 | 0.000 | 84.000..84.000 |
| val_core20_top1 | 0.817 | 0.048 | 0.762..0.857 |
| val_new_words_correct_commit_recall | 0.525 | 0.099 | 0.397..0.596 |
| val_new_words_macro_top1 | 0.802 | 0.020 | 0.782..0.822 |
| val_new_words_n_samples | 136.000 | 0.000 | 136.000..136.000 |
| val_new_words_top1 | 0.799 | 0.015 | 0.787..0.816 |
| val_weak6_correct_commit_recall | 0.234 | 0.169 | 0.091..0.429 |
| val_weak6_macro_top1 | 0.444 | 0.139 | 0.264..0.542 |
| val_weak6_n_samples | 21.667 | 0.500 | 21.000..22.000 |
| val_weak6_top1 | 0.447 | 0.136 | 0.273..0.545 |
| ps_test_correct_commit_recall | 0.451 | 0.156 | 0.263..0.575 |
| ps_test_n_hedge_refused | 7.667 | 5.000 | 2.000..12.000 |
| test_correct_commit_recall | 0.499 | 0.099 | 0.376..0.573 |
| test_n_hedge_refused | 1.667 | 1.500 | 0.000..3.000 |
| val_correct_commit_recall | 0.520 | 0.109 | 0.377..0.596 |
| val_n_hedge_refused | 1.333 | 1.500 | 0.000..3.000 |
| ps_test_swap_good_thanks | 8.667 | 1.000 | 8.000..10.000 |
| ps_test_swap_happy_please | 1.667 | 0.500 | 1.000..2.000 |
| test_swap_good_thanks | 6.000 | 3.000 | 3.000..9.000 |
| test_swap_happy_please | 1.000 | 1.000 | 0.000..2.000 |
| val_swap_good_thanks | 2.333 | 0.500 | 2.000..3.000 |
| val_swap_happy_please | 0.333 | 0.500 | 0.000..1.000 |


Two precision numbers appear in the tables: "typed precision, real words only" scores the gate's commits on clips that really are vocabulary words, while "typed precision, everything typed" also counts the commits it makes over REST/OTHER clips - all of which are wrong. These are offline commit-policy measurements, not a promise of live segmentation or sentence accuracy.

Each row is the mean over the seeds of one config, with its half-range. Configs are chosen on the val columns only; the test columns are reported, never used to decide. Comparisons require explicit matching label-set and evaluation identities; unknown historical identities do not qualify.

## What to fix

- Recommendations below use VAL only. Test/ps_test remain final, fixed-policy measurements, not configuration-selection inputs.
- VAL only, fixed p1 >= 0.85, margin >= 0.15, with saved pair/hedge rules: word commit share 63.6%, correct-commit recall 58.6%, commit precision 77.7%. Any future policy selection must use validation only, then be frozen before final test scoring. These are offline clips, not live accuracy.
- Hand detection over the SIGNING WINDOW averages 0.89 (clip-wide it is 0.44): the low clip number is the hands resting between signs, not a detection failure, and it is not the ceiling.
- MediaPipe is NOT what limits happy, like, now, please, thanks, wait, when: the hand is visible in over 80% of their signing window and the model still misses them (happy 25%, like 33%, now 33%, please 50%, thanks 33%, wait 0%, when 50%). That is a features-and-data problem - the segment read-out in features v3 and more signers - not camera framing and not more epochs.
- Genuinely hard to see in under 75% of the window: come, sorry, hungry, doctor, me, you, hello, your - framing helps these, and only these.
- VAL 62.5% is an offline result only. Check core20 retention and new-word coverage separately; it does not establish live accuracy.
- Nothing was re-run to produce this page. Every number came from the files training wrote.

## Files

- words_summary: D:\sign language project\ASTL project\ASLT\Model\videoModel\reports\words_summary.json
- meta: D:\sign language project\ASTL project\ASLT\Model\videoModel\models\meta.json
- per_word: D:\sign language project\ASTL project\ASLT\Model\videoModel\reports\per_word.csv
- confused_pairs: D:\sign language project\ASTL project\ASLT\Model\videoModel\reports\confused_pairs.csv
- manifest: D:\sign language project\ASTL project\ASLT\Model\videoModel\cache\manifest.csv

Trained on ASL Citizen (use.txt: non-commercial research use) plus 2520 PopSign v2.1 clips (CC BY 4.0). No webcam frame was used for training or calibration, and the live demo writes nothing to disk.
