# Word recognizer - results

Written by check_results.py on 20261118-225254. Read-only: it re-ran nothing.

## Headline

| | |
|---|---|
| test top-1 (official signer-disjoint) | 69.5% |
| test top-5 | 89.2% |
| test macro over words | 88.6% |
| test top-1 with REST/OTHER removed from the argmax | 90.0% |
| words-only top-1: train vs val (memorisation gap) | 95.4% vs 88.1% |
| best val top-1 | 65.9% |
| test size | 869 samples / 569 videos / 11 signers |
| train size | 1700 samples / 91 signers |
| epochs | 59 (best val 65.9%) |
| features | 200 dims, blocks all, 64 frames |
| classes | 22 |
| parameters | 378230 |
| seed / seconds | 20261116 / 356.5 (5.94 s per epoch) |

Every test number above is from signers the model never saw. A random split would read far higher and would not be reported here.

## Held-out groups (never pooled)

| group | videos | top-1 | words-only top-1 |
|---|---|---|---|
| ASL Citizen test signers (the live-app number) | 569 | 69.5% | 90.0% |
| PopSign val+test signers (second, independent style) | 120 | 84.2% | 84.2% |

Training saw 180 PopSign clips on top of the ASL Citizen train set. Each row is the signer-disjoint split of its own corpus; a number that pooled them would be meaningless and is not reported. The claim about the live webcam stays tied to the ASL Citizen row.

## Signer-disjointness check

- no problem found: no (source, signer) pair appears in train and in a held-out split

## Background classes

| | |
|---|---|
| REST typed as a word | 5.0% |
| OTHER typed as a word | 69.3% |
| a real word typed as REST/OTHER | 2.6% |

## Weakest words

- thanks 25.0% (12 samples, 11 signers)
- happy 41.7% (12 samples, 11 signers)
- please 76.9% (13 samples, 11 signers)
- good 83.3% (12 samples, 11 signers)
- eat 88.5% (26 samples, 11 signers)
- love 91.7% (12 samples, 11 signers)

## Most confused pairs

- good / thanks: 6 swapped of 24 (25.0%)
- thanks / bad: 3 swapped of 25 (12.0%)
- happy / please: 3 swapped of 25 (12.0%)
- good / happy: 2 swapped of 24 (8.3%)
- eat / thanks: 2 swapped of 38 (5.3%)
- love / yes: 1 swapped of 22 (4.5%)
- sad / happy: 1 swapped of 24 (4.2%)
- good / help: 1 swapped of 25 (4.0%)

## Hand detection per word (worst window rate first)

| word | signing window | whole clip | videos |
|---|---|---|---|
| sorry | 0.663 | 0.503 | 30 |
| good | 0.788 | 0.399 | 131 |
| bad | 0.802 | 0.335 | 30 |
| love | 0.844 | 0.406 | 30 |
| please | 0.853 | 0.480 | 31 |
| thanks | 0.855 | 0.385 | 131 |
| happy | 0.863 | 0.458 | 131 |
| hello | 0.872 | 0.398 | 32 |
| home | 0.876 | 0.413 | 31 |
| drink | 0.900 | 0.412 | 61 |
| OTHER | 0.902 | 0.454 | 1050 |
| yes | 0.902 | 0.423 | 30 |
| school | 0.903 | 0.456 | 30 |
| friend | 0.904 | 0.456 | 29 |
| eat | 0.906 | 0.400 | 63 |
| bye | 0.913 | 0.437 | 30 |
| help | 0.939 | 0.447 | 31 |
| no | 0.942 | 0.460 | 30 |
| sad | 0.947 | 0.433 | 30 |
| name | 0.958 | 0.433 | 32 |
| family | 0.965 | 0.482 | 30 |

The window column is the one to read: it is measured between sign_start and sign_end, where the sign actually happens. The clip column is what the extractor reports for the whole sentence clip, idle head and tail included. A word can only be missed because of detection if the window column is low - and per reports/predictions.csv it usually is not.

## Extraction errors

- none recorded

## Seed sweep: baseline vs new

**none-20261117-0843** - 3 seed(s), reg=none, feature v3, promoted seed 20261116

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.909 | 0.012 | 0.893..0.917 |
| test_words_top1 | 0.853 | 0.019 | 0.829..0.866 |
| test_typed_share | 0.737 | 0.011 | 0.725..0.747 |
| test_typed_precision | 0.731 | 0.018 | 0.717..0.752 |
| test_other_slip | 0.173 | 0.010 | 0.167..0.187 |
| test_rest_slip | 0.016 | 0.002 | 0.013..0.017 |
| test_swap_good_thanks | 8.000 | 0.000 | 8.000..8.000 |
| test_swap_happy_please | 4.667 | 0.500 | 4.000..5.000 |
| val_swap_good_thanks | 1.667 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.333 | 1.000 | 1.000..3.000 |

**none-20261117-0922** - 3 seed(s), reg=none, feature v3, promoted seed 20261117

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.909 | 0.012 | 0.893..0.917 |
| test_words_top1 | 0.853 | 0.019 | 0.829..0.866 |
| test_typed_share | 0.737 | 0.011 | 0.725..0.747 |
| test_typed_precision | 0.731 | 0.018 | 0.717..0.752 |
| test_other_slip | 0.173 | 0.010 | 0.167..0.187 |
| test_rest_slip | 0.016 | 0.002 | 0.013..0.017 |
| test_swap_good_thanks | 8.000 | 0.000 | 8.000..8.000 |
| test_swap_happy_please | 4.667 | 0.500 | 4.000..5.000 |
| val_swap_good_thanks | 1.667 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.333 | 1.000 | 1.000..3.000 |

**fair-20261117-1035-v3** - 3 seed(s), reg=none, feature v3, promoted seed none

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.909 | 0.012 | 0.893..0.917 |
| test_words_top1 | 0.853 | 0.019 | 0.829..0.866 |
| test_typed_share | 0.737 | 0.011 | 0.725..0.747 |
| test_typed_precision | 0.731 | 0.018 | 0.717..0.752 |
| test_other_slip | 0.173 | 0.010 | 0.167..0.187 |
| test_rest_slip | 0.016 | 0.002 | 0.013..0.017 |
| test_swap_good_thanks | 8.000 | 0.000 | 8.000..8.000 |
| test_swap_happy_please | 4.667 | 0.500 | 4.000..5.000 |
| val_swap_good_thanks | 1.667 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.333 | 1.000 | 1.000..3.000 |

**fair-20261117-1035-v2** - 3 seed(s), reg=none, feature v2, promoted seed none

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.893 | 0.012 | 0.881..0.905 |
| test_words_top1 | 0.890 | 0.022 | 0.863..0.907 |
| test_typed_share | 0.746 | 0.022 | 0.729..0.773 |
| test_typed_precision | 0.726 | 0.060 | 0.670..0.791 |
| test_other_slip | 0.193 | 0.062 | 0.127..0.250 |
| test_rest_slip | 0.026 | 0.007 | 0.017..0.030 |
| test_swap_good_thanks | 7.667 | 0.500 | 7.000..8.000 |
| test_swap_happy_please | 3.333 | 1.500 | 2.000..5.000 |
| val_swap_good_thanks | 1.333 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.667 | 0.500 | 2.000..3.000 |

**fair-20261117-1117-v3** - 3 seed(s), reg=none, feature v3, promoted seed none

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.909 | 0.012 | 0.893..0.917 |
| test_words_top1 | 0.853 | 0.019 | 0.829..0.866 |
| test_typed_share | 0.737 | 0.011 | 0.725..0.747 |
| test_typed_precision | 0.731 | 0.018 | 0.717..0.752 |
| test_other_slip | 0.173 | 0.010 | 0.167..0.187 |
| test_rest_slip | 0.016 | 0.002 | 0.013..0.017 |
| test_swap_good_thanks | 8.000 | 0.000 | 8.000..8.000 |
| test_swap_happy_please | 4.667 | 0.500 | 4.000..5.000 |
| val_swap_good_thanks | 1.667 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.333 | 1.000 | 1.000..3.000 |

**fair-20261117-1117-v2** - 3 seed(s), reg=none, feature v2, promoted seed 20261116

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.893 | 0.012 | 0.881..0.905 |
| test_words_top1 | 0.890 | 0.022 | 0.863..0.907 |
| test_typed_share | 0.746 | 0.022 | 0.729..0.773 |
| test_typed_precision | 0.726 | 0.060 | 0.670..0.791 |
| test_other_slip | 0.193 | 0.062 | 0.127..0.250 |
| test_rest_slip | 0.026 | 0.007 | 0.017..0.030 |
| test_swap_good_thanks | 7.667 | 0.500 | 7.000..8.000 |
| test_swap_happy_please | 3.333 | 1.500 | 2.000..5.000 |
| val_swap_good_thanks | 1.333 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.667 | 0.500 | 2.000..3.000 |

**popsign60** - 3 seed(s), reg=none, feature v3, promoted seed 20261118

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.889 | 0.012 | 0.881..0.905 |
| test_words_top1 | 0.890 | 0.009 | 0.877..0.896 |
| test_typed_share | 0.742 | 0.035 | 0.714..0.784 |
| test_typed_precision | 0.792 | 0.008 | 0.781..0.797 |
| test_other_slip | 0.138 | 0.010 | 0.127..0.147 |
| test_rest_slip | 0.011 | 0.005 | 0.007..0.017 |
| test_swap_good_thanks | 6.000 | 1.500 | 5.000..8.000 |
| test_swap_happy_please | 3.333 | 1.500 | 2.000..5.000 |
| val_swap_good_thanks | 1.333 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.667 | 0.500 | 2.000..3.000 |

**popsign60full** - 3 seed(s), reg=none, feature v3, promoted seed 20261116

| metric | mean | spread (half-range) | min..max |
|---|---|---|---|
| val_words_top1 | 0.893 | 0.018 | 0.881..0.917 |
| test_words_top1 | 0.900 | 0.011 | 0.888..0.911 |
| test_typed_share | 0.720 | 0.022 | 0.695..0.740 |
| test_typed_precision | 0.834 | 0.020 | 0.812..0.853 |
| test_other_slip | 0.111 | 0.020 | 0.093..0.133 |
| test_rest_slip | 0.010 | 0.003 | 0.007..0.013 |
| ps_test_swap_good_thanks | 15.000 | 4.500 | 11.000..20.000 |
| ps_test_swap_happy_please | 0.000 | 0.000 | 0.000..0.000 |
| test_swap_good_thanks | 6.333 | 1.500 | 5.000..8.000 |
| test_swap_happy_please | 4.000 | 1.000 | 3.000..5.000 |
| val_swap_good_thanks | 1.333 | 0.500 | 1.000..2.000 |
| val_swap_happy_please | 2.667 | 0.500 | 2.000..3.000 |

## Does it beat the baseline? (decided on val_top1)

The baseline's own seed-to-seed spread is +-0.014. A gain smaller than that is noise.

| run | val top-1 | vs baseline | bigger than the spread? |
|---|---|---|---|
| none-20261117-0843 | 0.541 | +0.000 | baseline |
| none-20261117-0922 | 0.541 | +0.000 | no - inside the noise |
| fair-20261117-1035-v3 | 0.541 | +0.000 | no - inside the noise |
| fair-20261117-1035-v2 | 0.552 | +0.011 | no - inside the noise |
| fair-20261117-1117-v3 | 0.541 | +0.000 | no - inside the noise |
| fair-20261117-1117-v2 | 0.552 | +0.011 | no - inside the noise |
| popsign60 | 0.621 | +0.080 | YES - adopt |
| popsign60full | 0.653 | +0.112 | YES - adopt |


## Fair comparison: feature v3 (fair-20261117-1117-v3) vs v2 (fair-20261117-1117-v2)

| metric | v3 mean | v3 spread | v2 mean | v2 spread |
|---|---|---|---|---|
| val top-1 (DECIDES) | 0.541 | +-0.014 | 0.552 | +-0.025 |
| words-only test top-1 | 0.853 | +-0.019 | 0.890 | +-0.022 |
| words-only test top-5 | 0.975 | +-0.006 | 0.979 | +-0.007 |
| typed share at the gate | 0.737 | +-0.011 | 0.746 | +-0.022 |
| typed precision, real words only | 0.940 | +-0.025 | 0.959 | +-0.005 |
| typed precision, everything typed | 0.731 | +-0.018 | 0.726 | +-0.060 |
| OTHER slip | 0.173 | +-0.010 | 0.193 | +-0.062 |
| REST slip | 0.016 | +-0.002 | 0.026 | +-0.007 |
| good/thanks swapped | 8.000 | +-0.000 | 7.667 | +-0.500 |
| happy/please swapped | 4.667 | +-0.500 | 3.333 | +-1.500 |

val decides: v3 0.541 vs v2 0.552 (diff -0.011, pooled seed spread +-0.026) -> keep v2 - v3 is inside the seed spread or behind.


Two precision numbers appear in the tables: "typed precision, real words only" scores the gate's commits on clips that really are vocabulary words, while "typed precision, everything typed" also counts the commits it makes over REST/OTHER clips - all of which are wrong - so the second is never higher, and the second is what the live demo actually delivers.

Each row is the mean over the seeds of one config, with its half-range. Configs are chosen on the val columns only; the test columns are reported, never used to decide. A gain smaller than the baseline's own spread is not a gain.

## What to fix

- OTHER is typed as a word 69.3% of the time, and that is not a bug worth chasing: OTHER is ~1000 unrelated signs, not a category, so no network can learn it as one class. Rejection has to come from the confidence gate, never from the OTHER logit. Measured on these readings: committing only at p1 >= 0.80 still types 74% of real words, and the words it does type are right 99% of the time (OTHER slips through 13%, REST 1%). That number, not the raw top-1, is what the demo should promise.
- Hand detection over the SIGNING WINDOW averages 0.88 (clip-wide it is 0.43): the low clip number is the hands resting between signs, not a detection failure, and it is not the ceiling.
- MediaPipe is NOT what limits happy, thanks: the hand is visible in over 80% of their signing window and the model still misses them (happy 42%, thanks 25%). That is a features-and-data problem - the segment read-out in features v3 and more signers - not camera framing and not more epochs.
- Genuinely hard to see in under 75% of the window: sorry - framing helps these, and only these.
- 69.5% is a number worth building on. Try the live demo and the guided drill - that is where the remaining problems show up.
- Nothing was re-run to produce this page. Every number came from the files training wrote.

## Files

- words_summary: D:\sign language project\ASTL project\ASLT\videoModel\reports\words_summary.json
- meta: D:\sign language project\ASTL project\ASLT\videoModel\models\meta.json
- per_word: D:\sign language project\ASTL project\ASLT\videoModel\reports\per_word.csv
- confused_pairs: D:\sign language project\ASTL project\ASLT\videoModel\reports\confused_pairs.csv
- manifest: D:\sign language project\ASTL project\ASLT\videoModel\cache\manifest.csv

Trained on ASL Citizen (use.txt: non-commercial research use) plus 180 PopSign v2.1 clips (CC BY 4.0). No webcam frame was used for training or calibration, and the live demo writes nothing to disk.
