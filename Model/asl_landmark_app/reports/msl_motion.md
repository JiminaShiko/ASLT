# J/Z as video classes: what was trained, and why it is not shipped

Measured 2026-11-15 on the MSL dynamic corpus (Zenodo 14689869, "Mexican Sign
Language Alphabet - dynamic signs only", CC-BY-4.0, 622 frontal clips, letters
J K Ñ Q X Z, 18 train signers + 2 disjoint test signers). The clips were
unpacked once, run through MediaPipe VIDEO landmarks, and cached; the archive
and the mp4s were then deleted with the user's approval, so this file plus
`cache/msl_landmarks*` and `cache/msl_windows.pkl` is what remains.

Reproduce the surviving part with `python src/verify_jz_trigger.py` (the
precondition measurement, which reads the landmark cache). The trained head
below was measured by throwaway scripts, deliberately not added to `src/`: it
is not shipped, and a script that trains an unsold model is a trap for the next
reader.

## Corpus facts that matter

| fact | value |
|---|---|
| clips | 622 (562 train / 18 signers, 60 test / S20 + S21, zero overlap) |
| landmark frames cached | 50,022 |
| MediaPipe VIDEO detection rate per clip | median 100 %, p10 96 %, 3 clips < 50 % |
| cost | 11.6-17 ms per frame |
| median palm (wrist to middle-MCP) | 0.129 normalised image units |
| frame rate | 30.0 or ~58-60 fps, so windows are matched in **seconds**, not frames |
| windows built | MSL 1894 (J 253, K 306, Q 293, X 352, Z 332, Ñ 358), local ASL 2800 |

## The features

102-dim motion vector per 1.2 s window: 20 resampled positions each for wrist-x,
wrist-y, index-tip-x, index-tip-y (80), then 22 scalars - path length and net
displacement per stream (8), direction-reversal counts (4, plus 2 on a shorter
sub-window), mean/range wrist-y and tip-y, mean/std finger extension, the share
of frames passing the shape gate, mean/std palm size. Learner:
`StandardScaler -> MLPClassifier(64)`, `early_stopping`, seed 0. Combined
"motion + static" is those 102 plus the shipped 71 = 173.

Everything is computed on `normalize.canonicalize`d landmarks through the same
helpers the live decider uses, so the offline series and the live series are the
same quantity.

## Results

Headline folds are **leave-one-signer-out** over the 18 MSL signers. A random
`KFold(shuffle=True)` split was also measured and is leaky by construction -
neighbouring windows of one clip land on both sides - so it is quoted nowhere as
an accuracy.

3-class head (J / Z / neither), precision on held-out MSL signers:

| features | J | Z | neither |
|---|---|---|---|
| motion only | **91.9 %** (137/149) | **85.5 %** (212/248) | 87.0 % (1149/1321) |
| motion + static | **98.1 %** (209/213) | **93.1 %** (270/290) | 95.5 % (1160/1215) |

Recall on the same folds: motion-only J 90.0 %, Z 46.7 %; motion+static J 95.0 %,
Z 36.7 % (Z is precise but timid).

Against the hand-written rule on the identical windows: the rule fires on 22.4 %
of MSL train windows and 30.7 % of test windows, and it is **not
discriminative** - it fires on Q and X about as often as on J. Restricted to the
shape-gated subset it names J on 27.1 % (train) / 22.1 % (test) and Z on
0.2 % / 0.0 %, and stays silent on 72.6 % / 77.9 %.

## The transfer test, which is the whole point

Trained on MSL, applied unchanged to 1,499 local ASL windows (grassknoted
J/Z/I/S/A/N/M, 400 frames per letter for J, Z, I, S, A, N, M):

| features | window | J recall | Z recall | neither recall | accuracy (chance) |
|---|---|---|---|---|---|
| motion | MSL unseen signers | 90.0 % | 46.7 % | 86.5 % | 80.1 % (71.6 %) |
| motion | **local ASL** | **0.8 %** | 2.2 % | 98.4 % | 46.7 % (46.6 %) |
| both | MSL unseen signers | 95.0 % | 36.7 % | 89.7 % | 81.2 % (71.6 %) |
| both | **local ASL** | **1.2 %** | 41.8 % | 98.3 % | 57.3 % (46.6 %) |

On the user's own language the head is indistinguishable from always answering
"neither". **Not shipped.**

## Why it fails, measured rather than assumed

The local frames labelled J and Z barely move. Wrist path over a 1.2 s window,
in palm units:

| stream | wrist path |
|---|---|
| MSL J | 9.08 |
| local ASL J | 1.25 |
| local ASL I | 1.19 |
| local ASL N | 11.10 |
| local ASL M | 12.93 |

So J and I differ by 0.06 palm units of movement while N moves nine times more -
the photographs labelled J are an I-shaped hand with a curled index, and the
movement that defines the letter was never in the data. That is also the
explanation for the 51.7 % J-vs-I ceiling measured earlier on static features:
no feature set separates two letters whose difference was not recorded.

A trained motion head cannot invent the movement, and the brief forbids
recording the user's own webcam, so there is no ASL motion corpus to train on
here. That is the actual constraint, stated plainly.

## What was shipped instead

The same measurement session found the thing that does generalise: the deployed
24-class model already knows J and Z by their *closest static letter* - true J
is displayed as I on 92 % of frames, true Z as X on 86 %, and a frame whose
top-1 is X with `p1 < 0.90` is 93 % truly Z. That reading is now the second
precondition in front of the unchanged trajectory rule; the per-letter numbers,
the code diff description and the limits of the evidence are in
`reports/jz_trigger.md`.
