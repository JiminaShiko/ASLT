# Word recognizer - offline results
Offline held-out clip evaluation, not measured live/webcam accuracy.

Written by check_results.py on 20261128-124912. Read-only: it re-ran nothing.

## Headline

| | |
|---|---|
| test top-1 (official signer-disjoint) | 83.7% |
| test top-5 | 96.4% |
| test macro over words | 82.8% |
| test top-1 with REST/OTHER removed from the argmax | 83.7% |
| words-only top-1: train vs val (memorisation gap) | 95.5% vs 90.2% |
| best val top-1 | 87.2% |
| test size | 860 samples / 565 videos / 11 signers |
| train size | 3070 samples / 90 signers |
| epochs | 47 (best val 87.2%) |
| features | 200 dims, blocks all, 64 frames |
| classes | 42 |
| label-set identity | 437a28cca1a2f2f960ff |
| evaluation identity | c969703fbf68a7799b04 |
| parameters | 382090 |
| seed / seconds | 20261117 / 521 (10.89 s per epoch) |

Every test number above is from signers the model never saw. A random split would read far higher and would not be reported here.

## The shipped artifact is an ensemble
`models/words_core20.pt` plus `words_core20_e2.pt`, `words_core20_e3.pt`: the number typed is the mean of their probabilities. Every row below is the same cached held-out clips scored through each checkpoint, so the columns are comparable.

| model | val words | test words | test macro words | ps_test words |
|---|---|---|---|---|
| ensemble (shipped) | 90.2% | 84.8% | 84.8% | 83.8% |
| single: reg | 89.6% | 82.8% | 82.8% | 80.5% |
| single: none | 88.3% | 83.9% | 83.7% | 84.3% |
| single: aug+ | 87.7% | 83.0% | 83.0% | 83.0% |

Members were trained on the training splits only and the shipped combination was chosen on VAL.

## Held-out groups (never pooled)

| group | videos | top-1 | words-only top-1 |
|---|---|---|---|
| ASL Citizen official test signers (offline) | 565 | 83.7% | 83.7% |
| PopSign val+test signers (second, independent style) | 600 | 80.5% | 80.5% |

Training saw 1800 PopSign clips on top of the ASL Citizen train set. Each row is the signer-disjoint split of its own corpus; a number that pooled them would be meaningless and is not reported. Neither row is a measurement of live webcam accuracy.

## Signer-disjointness check

- no problem found: no (source, signer) pair appears in train and in a held-out split

## Retention and expansion (full-classifier argmax)

Core20 is active batch 1; new_words is the active later batches. Expanded-total accuracy is not an improvement over an old smaller-vocabulary baseline.
Weak-six membership: six weakest core20 words on this run's official val split; fixed from VAL for every final split.

| split | group | samples | words scored | top-1 | macro | correct-commit recall | labels |
|---|---|---|---|---|---|---|---|
| val | core20 | 76 | 18 | 92.1% | 91.0% | 68.4% | eat, family, friend, good, hello, help, name, school, drink, thanks, love, bye, sorry, yes, no, please, bad, home |
| val | weak6 | 23 | 6 | 73.9% | 73.1% | 65.2% | thanks, bad, hello, name, love, bye |
| val | new_words | 87 | 22 | 87.4% | 89.1% | 58.6% | me, you, what, where, want, need, go, come, stop, more, water, understand, now, today, work, finish, sleep, bathroom, hungry, when, why, how |
| test | core20 | 245 | 18 | 85.3% | 84.0% | 57.6% | eat, family, friend, good, hello, help, name, school, drink, thanks, love, bye, sorry, yes, no, please, bad, home |
| test | weak6 | 77 | 6 | 79.2% | 78.7% | 50.6% | thanks, bad, hello, name, love, bye |
| test | new_words | 302 | 22 | 80.8% | 81.7% | 53.3% | me, you, what, where, want, need, go, come, stop, more, water, understand, now, today, work, finish, sleep, bathroom, hungry, when, why, how |
| ps_test | core20 | 280 | 14 | 81.1% | 81.1% | 48.6% | eat, family, friend, good, hello, help, name, school, drink, thanks, love, bye, sorry, yes, no, please, bad, home |
| ps_test | weak6 | 100 | 5 | 72.0% | 72.0% | 34.0% | thanks, bad, hello, name, love, bye |
| ps_test | new_words | 320 | 16 | 80.0% | 80.0% | 57.5% | me, you, what, where, want, need, go, come, stop, more, water, understand, now, today, work, finish, sleep, bathroom, hungry, when, why, how |

## Fixed commit policy (offline clips)

Policy is saved with this run; test is not re-thresholded by this report. Missing historical policy metrics are not reconstructed from rounded CSVs.
Saved policy: `{"confidence_min": 0.85, "hedge_pairs": {"good|thanks": "good/thanks?", "me|sorry": "sorry/me?", "now|today": "now/today?"}, "margin_min": 0.15, "pair_margins": {}, "scope": "offline clip decisions; not measured live accuracy", "version": "predictor.decide/commitable-v1"}`

| split | commits | precision | word commit share | correct-commit recall | hedge refusals | REST slip | OTHER slip |
|---|---|---|---|---|---|---|---|
| val | 112 | 92.0% | 65.6% | 63.2% | 4 | 2.7% | 14.3% |
| test | 325 | 92.9% | 57.8% | 55.2% | 13 | 1.3% | 38.5% |
| ps_test | 325 | 98.5% | 54.2% | 53.3% | 30 | n/a | n/a |

## Background classes (raw argmax, before commit policy)

| | |
|---|---|
| REST predicted as a word | 11.0% |
| OTHER predicted as a word | 92.3% |
| a real word predicted as REST/OTHER | 2.4% |

## Weakest words (test description only, not selection)

- thanks 50.0% (12 samples, 11 signers)
- what 56.0% (25 samples, 11 signers)
- good 58.3% (12 samples, 11 signers)
- bad 61.5% (13 samples, 11 signers)
- me 61.5% (13 samples, 11 signers)
- need 66.7% (12 samples, 11 signers)

## Most confused pairs

- what / want: 8 swapped of 49 (16.3%)
- good / thanks: 7 swapped of 24 (29.2%)
- sorry / me: 6 swapped of 25 (24.0%)
- now / today: 4 swapped of 23 (17.4%)
- love / sorry: 3 swapped of 24 (12.5%)
- you / need: 3 swapped of 24 (12.5%)
- what / finish: 3 swapped of 39 (7.7%)
- friend / when: 2 swapped of 24 (8.3%)

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

**reg-20261128-1213** - 3 seed(s), reg=reg, feature v3, promoted seed 20261117

label_set_id: 437a28cca1a2f2f960ff
evaluation_id: c969703fbf68a7799b04

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.883 | 0.021 | 0.859..0.902 |
| val_top1 | 0.862 | 0.009 | 0.853..0.872 |
| val_typed_share | 0.675 | 0.025 | 0.656..0.706 |
| val_typed_precision | 0.931 | 0.010 | 0.920..0.939 |
| val_other_slip | 0.191 | 0.071 | 0.143..0.286 |
| val_rest_slip | 0.029 | 0.003 | 0.027..0.033 |
| test_words_top1 | 0.848 | 0.009 | 0.837..0.856 |
| test_words_top5 | 0.979 | 0.006 | 0.974..0.985 |
| test_top1 | 0.841 | 0.005 | 0.837..0.847 |
| test_macro_words | 0.840 | 0.011 | 0.828..0.849 |
| test_typed_share | 0.617 | 0.032 | 0.578..0.642 |
| test_typed_precision | 0.930 | 0.001 | 0.929..0.930 |
| test_typed_precision_words | 0.956 | 0.007 | 0.949..0.962 |
| test_other_slip | 0.359 | 0.038 | 0.308..0.385 |
| test_rest_slip | 0.016 | 0.007 | 0.010..0.023 |
| test_other_as_word | 0.923 | 0.077 | 0.846..1.000 |
| ps_test_words_top1 | 0.820 | 0.012 | 0.805..0.828 |
| ps_test_top1 | 0.820 | 0.012 | 0.805..0.828 |
| ps_test_typed_share | 0.571 | 0.030 | 0.542..0.602 |
| ps_test_typed_precision_words | 0.965 | 0.019 | 0.947..0.985 |
| ps_test_core20_correct_commit_recall | 0.521 | 0.030 | 0.486..0.546 |
| ps_test_core20_macro_top1 | 0.824 | 0.013 | 0.811..0.836 |
| ps_test_core20_n_samples | 280.000 | 0.000 | 280.000..280.000 |
| ps_test_core20_top1 | 0.824 | 0.013 | 0.811..0.836 |
| ps_test_new_words_correct_commit_recall | 0.575 | 0.016 | 0.559..0.591 |
| ps_test_new_words_macro_top1 | 0.817 | 0.016 | 0.800..0.831 |
| ps_test_new_words_n_samples | 320.000 | 0.000 | 320.000..320.000 |
| ps_test_new_words_top1 | 0.817 | 0.016 | 0.800..0.831 |
| ps_test_weak6_correct_commit_recall | 0.415 | 0.095 | 0.340..0.530 |
| ps_test_weak6_macro_top1 | 0.747 | 0.060 | 0.700..0.820 |
| ps_test_weak6_n_samples | 93.333 | 10.000 | 80.000..100.000 |
| ps_test_weak6_top1 | 0.747 | 0.060 | 0.700..0.820 |
| test_core20_correct_commit_recall | 0.601 | 0.020 | 0.576..0.616 |
| test_core20_macro_top1 | 0.869 | 0.031 | 0.840..0.903 |
| test_core20_n_samples | 245.000 | 0.000 | 245.000..245.000 |
| test_core20_top1 | 0.875 | 0.024 | 0.853..0.902 |
| test_new_words_correct_commit_recall | 0.579 | 0.036 | 0.533..0.606 |
| test_new_words_macro_top1 | 0.816 | 0.010 | 0.806..0.825 |
| test_new_words_n_samples | 302.000 | 0.000 | 302.000..302.000 |
| test_new_words_top1 | 0.809 | 0.012 | 0.798..0.821 |
| test_weak6_correct_commit_recall | 0.562 | 0.067 | 0.506..0.640 |
| test_weak6_macro_top1 | 0.797 | 0.038 | 0.764..0.839 |
| test_weak6_n_samples | 75.333 | 1.500 | 74.000..77.000 |
| test_weak6_top1 | 0.801 | 0.032 | 0.773..0.838 |
| val_core20_correct_commit_recall | 0.702 | 0.013 | 0.684..0.711 |
| val_core20_macro_top1 | 0.896 | 0.017 | 0.876..0.910 |
| val_core20_n_samples | 76.000 | 0.000 | 76.000..76.000 |
| val_core20_top1 | 0.912 | 0.013 | 0.895..0.921 |
| val_new_words_correct_commit_recall | 0.625 | 0.040 | 0.586..0.667 |
| val_new_words_macro_top1 | 0.863 | 0.035 | 0.821..0.891 |
| val_new_words_n_samples | 87.000 | 0.000 | 87.000..87.000 |
| val_new_words_top1 | 0.851 | 0.029 | 0.816..0.874 |
| val_weak6_correct_commit_recall | 0.604 | 0.064 | 0.524..0.652 |
| val_weak6_macro_top1 | 0.695 | 0.039 | 0.653..0.731 |
| val_weak6_n_samples | 22.000 | 1.000 | 21.000..23.000 |
| val_weak6_top1 | 0.711 | 0.036 | 0.667..0.739 |
| ps_test_correct_commit_recall | 0.550 | 0.018 | 0.533..0.570 |
| ps_test_n_hedge_refused | 38.333 | 11.000 | 30.000..52.000 |
| test_correct_commit_recall | 0.589 | 0.028 | 0.552..0.609 |
| test_n_hedge_refused | 19.000 | 4.500 | 13.000..22.000 |
| val_correct_commit_recall | 0.661 | 0.028 | 0.632..0.687 |
| val_n_hedge_refused | 4.000 | 1.000 | 3.000..5.000 |
| ps_test_swap_good_thanks | 8.333 | 1.500 | 7.000..10.000 |
| ps_test_swap_happy_please | 0.000 | 0.000 | 0.000..0.000 |
| test_swap_good_thanks | 7.667 | 3.000 | 5.000..11.000 |
| test_swap_happy_please | 0.000 | 0.000 | 0.000..0.000 |
| val_swap_good_thanks | 2.000 | 0.000 | 2.000..2.000 |
| val_swap_happy_please | 0.000 | 0.000 | 0.000..0.000 |


Two precision numbers appear in the tables: "typed precision, real words only" scores the gate's commits on clips that really are vocabulary words, while "typed precision, everything typed" also counts the commits it makes over REST/OTHER clips - all of which are wrong. These are offline commit-policy measurements, not a promise of live segmentation or sentence accuracy.

Each row is the mean over the seeds of one config, with its half-range. Configs are chosen on the val columns only; the test columns are reported, never used to decide. Comparisons require explicit matching label-set and evaluation identities; unknown historical identities do not qualify.

## What to fix

- Recommendations below use VAL only. Test/ps_test remain final, fixed-policy measurements, not configuration-selection inputs.
- VAL only, fixed p1 >= 0.85, margin >= 0.15, with saved pair/hedge rules: word commit share 65.6%, correct-commit recall 63.2%, commit precision 92.0%. Any future policy selection must use validation only, then be frozen before final test scoring. These are offline clips, not live accuracy.
- Hand detection over the SIGNING WINDOW averages 0.89 (clip-wide it is 0.44): the low clip number is the hands resting between signs, not a detection failure, and it is not the ceiling.
- MediaPipe is NOT what limits thanks, want: the hand is visible in over 80% of their signing window and the model still misses them (thanks 33%, want 57%). That is a features-and-data problem - the segment read-out in features v3 and more signers - not camera framing and not more epochs.
- Genuinely hard to see in under 75% of the window: come, sorry, hungry, doctor, me, you, hello, your - framing helps these, and only these.
- VAL 87.2% is an offline result only. Check core20 retention and new-word coverage separately; it does not establish live accuracy.
- Nothing was re-run to produce this page. Every number came from the files training wrote.

## Files

- words_summary: D:\sign language project\ASTL project\ASLT\Model\videoModel\reports\words_summary.json
- meta: D:\sign language project\ASTL project\ASLT\Model\videoModel\models\meta.json
- per_word: D:\sign language project\ASTL project\ASLT\Model\videoModel\reports\per_word.csv
- confused_pairs: D:\sign language project\ASTL project\ASLT\Model\videoModel\reports\confused_pairs.csv
- manifest: D:\sign language project\ASTL project\ASLT\Model\videoModel\cache\manifest.csv

Trained on ASL Citizen (use.txt: non-commercial research use) plus 1800 PopSign v2.1 clips (CC BY 4.0). No webcam frame was used for training or calibration, and the live demo writes nothing to disk.
