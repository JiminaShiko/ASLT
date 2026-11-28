# Seed runs (mean +- spread over seeds of the same config)

Decided by the VAL columns. The TEST columns are printed because
you asked for them, not because they choose anything: picking a
config on test would make the test number meaningless.

Candidate: boost_reg (offline evaluation, not deployed).

Core20 retention and new-word totals are separate. Different or unknown
label/evaluation identities are not baseline improvements.

## reg-20261128-1213   (3 seed(s), reg=reg, feature v3)
label_set_id: 437a28cca1a2f2f960ff
evaluation_id: c969703fbf68a7799b04
    val_words_top1              0.883 +-0.021   (0.859..0.902)  n=3
    val_top1                    0.862 +-0.009   (0.853..0.872)  n=3
    val_typed_share             0.675 +-0.025   (0.656..0.706)  n=3
    val_typed_precision         0.931 +-0.010   (0.920..0.939)  n=3
    val_other_slip              0.191 +-0.071   (0.143..0.286)  n=3
    val_rest_slip               0.029 +-0.003   (0.027..0.033)  n=3
    test_words_top1             0.848 +-0.009   (0.837..0.856)  n=3
    test_words_top5             0.979 +-0.006   (0.974..0.985)  n=3
    test_top1                   0.841 +-0.005   (0.837..0.847)  n=3
    test_macro_words            0.840 +-0.011   (0.828..0.849)  n=3
    test_typed_share            0.617 +-0.032   (0.578..0.642)  n=3
    test_typed_precision        0.930 +-0.001   (0.929..0.930)  n=3
    test_typed_precision_words  0.956 +-0.007   (0.949..0.962)  n=3
    test_other_slip             0.359 +-0.038   (0.308..0.385)  n=3
    test_rest_slip              0.016 +-0.007   (0.010..0.023)  n=3
    test_other_as_word          0.923 +-0.077   (0.846..1.000)  n=3
    ps_test_words_top1          0.820 +-0.012   (0.805..0.828)  n=3
    ps_test_top1                0.820 +-0.012   (0.805..0.828)  n=3
    ps_test_typed_share         0.571 +-0.030   (0.542..0.602)  n=3
    ps_test_typed_precision_words  0.965 +-0.019   (0.947..0.985)  n=3
    ps_test_core20_correct_commit_recall  0.521 +-0.030   (0.486..0.546)  n=3
    ps_test_core20_macro_top1   0.824 +-0.013   (0.811..0.836)  n=3
    ps_test_core20_n_samples    280.000 +-0.000   (280.000..280.000)  n=3
    ps_test_core20_top1         0.824 +-0.013   (0.811..0.836)  n=3
    ps_test_new_words_correct_commit_recall  0.575 +-0.016   (0.559..0.591)  n=3
    ps_test_new_words_macro_top1  0.817 +-0.016   (0.800..0.831)  n=3
    ps_test_new_words_n_samples  320.000 +-0.000   (320.000..320.000)  n=3
    ps_test_new_words_top1      0.817 +-0.016   (0.800..0.831)  n=3
    ps_test_weak6_correct_commit_recall  0.415 +-0.095   (0.340..0.530)  n=3
    ps_test_weak6_macro_top1    0.747 +-0.060   (0.700..0.820)  n=3
    ps_test_weak6_n_samples     93.333 +-10.000   (80.000..100.000)  n=3
    ps_test_weak6_top1          0.747 +-0.060   (0.700..0.820)  n=3
    test_core20_correct_commit_recall  0.601 +-0.020   (0.576..0.616)  n=3
    test_core20_macro_top1      0.869 +-0.031   (0.840..0.903)  n=3
    test_core20_n_samples       245.000 +-0.000   (245.000..245.000)  n=3
    test_core20_top1            0.875 +-0.024   (0.853..0.902)  n=3
    test_new_words_correct_commit_recall  0.579 +-0.036   (0.533..0.606)  n=3
    test_new_words_macro_top1   0.816 +-0.010   (0.806..0.825)  n=3
    test_new_words_n_samples    302.000 +-0.000   (302.000..302.000)  n=3
    test_new_words_top1         0.809 +-0.012   (0.798..0.821)  n=3
    test_weak6_correct_commit_recall  0.562 +-0.067   (0.506..0.640)  n=3
    test_weak6_macro_top1       0.797 +-0.038   (0.764..0.839)  n=3
    test_weak6_n_samples        75.333 +-1.500   (74.000..77.000)  n=3
    test_weak6_top1             0.801 +-0.032   (0.773..0.838)  n=3
    val_core20_correct_commit_recall  0.702 +-0.013   (0.684..0.711)  n=3
    val_core20_macro_top1       0.896 +-0.017   (0.876..0.910)  n=3
    val_core20_n_samples        76.000 +-0.000   (76.000..76.000)  n=3
    val_core20_top1             0.912 +-0.013   (0.895..0.921)  n=3
    val_new_words_correct_commit_recall  0.625 +-0.040   (0.586..0.667)  n=3
    val_new_words_macro_top1    0.863 +-0.035   (0.821..0.891)  n=3
    val_new_words_n_samples     87.000 +-0.000   (87.000..87.000)  n=3
    val_new_words_top1          0.851 +-0.029   (0.816..0.874)  n=3
    val_weak6_correct_commit_recall  0.604 +-0.064   (0.524..0.652)  n=3
    val_weak6_macro_top1        0.695 +-0.039   (0.653..0.731)  n=3
    val_weak6_n_samples         22.000 +-1.000   (21.000..23.000)  n=3
    val_weak6_top1              0.711 +-0.036   (0.667..0.739)  n=3
    ps_test_correct_commit_recall  0.550 +-0.018   (0.533..0.570)  n=3
    ps_test_n_hedge_refused     38.333 +-11.000   (30.000..52.000)  n=3
    test_correct_commit_recall  0.589 +-0.028   (0.552..0.609)  n=3
    test_n_hedge_refused        19.000 +-4.500   (13.000..22.000)  n=3
    val_correct_commit_recall   0.661 +-0.028   (0.632..0.687)  n=3
    val_n_hedge_refused         4.000 +-1.000   (3.000..5.000)  n=3
    ps_test_swap_good_thanks    8.333 +-1.500   (7.000..10.000)  n=3
    ps_test_swap_happy_please   0.000 +-0.000   (0.000..0.000)  n=3
    test_swap_good_thanks       7.667 +-3.000   (5.000..11.000)  n=3
    test_swap_happy_please      0.000 +-0.000   (0.000..0.000)  n=3
    val_swap_good_thanks        2.000 +-0.000   (2.000..2.000)  n=3
    val_swap_happy_please       0.000 +-0.000   (0.000..0.000)  n=3
