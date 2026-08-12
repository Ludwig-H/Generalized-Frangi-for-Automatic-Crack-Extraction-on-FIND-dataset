## Segmentation, par bras et par graine

| Variante | Graine | IoU | IoU@3px | Dice | Précision | Rappel | clDice | Composantes |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| A0 baseline | 13 | 0.6674 | 0.8405 | 0.7887 | 0.8252 | 0.7860 | 0.9000 | 1.7000 |
| A1 rgb_recalibration | 13 | 0.7112 | 0.8870 | 0.8226 | 0.8201 | 0.8517 | 0.9124 | 4.2000 |
| A1 rgb_recalibration | 37 | 0.7078 | 0.8835 | 0.8199 | 0.8203 | 0.8469 | 0.9089 | 3.7778 |
| A1 rgb_recalibration | 73 | 0.7097 | 0.8863 | 0.8213 | 0.8199 | 0.8504 | 0.9115 | 4.3000 |
| A2 frangi_signed_abstention | 13 | 0.7071 | 0.8880 | 0.8204 | 0.8215 | 0.8442 | 0.8902 | 16.3000 |
| A2 frangi_signed_abstention | 37 | 0.7069 | 0.8882 | 0.8201 | 0.8183 | 0.8478 | 0.8974 | 13.4000 |
| A2 frangi_signed_abstention | 73 | 0.7052 | 0.8866 | 0.8190 | 0.8165 | 0.8464 | 0.8981 | 10.7111 |
| A3 frangi_permuted | 13 | 0.7066 | 0.8863 | 0.8193 | 0.8246 | 0.8407 | 0.8960 | 16.8889 |
| A3 frangi_permuted | 37 | 0.7081 | 0.8865 | 0.8200 | 0.8143 | 0.8533 | 0.8850 | 21.9333 |
| A3 frangi_permuted | 73 | 0.7046 | 0.8842 | 0.8175 | 0.8190 | 0.8438 | 0.8953 | 16.4556 |
| A4 raw_thermal | 13 | 0.7067 | 0.8877 | 0.8198 | 0.8207 | 0.8448 | 0.9038 | 13.1778 |
| A4 raw_thermal | 37 | 0.7048 | 0.8861 | 0.8181 | 0.8224 | 0.8405 | 0.9040 | 11.1667 |
| A4 raw_thermal | 73 | 0.7033 | 0.8846 | 0.8170 | 0.8108 | 0.8511 | 0.8968 | 14.7000 |
| A5 frangi_no_abstention | 13 | 0.7050 | 0.8863 | 0.8191 | 0.8225 | 0.8415 | 0.8983 | 10.6111 |
| A5 frangi_no_abstention | 37 | 0.7037 | 0.8853 | 0.8181 | 0.8200 | 0.8422 | 0.8953 | 13.5222 |
| A5 frangi_no_abstention | 73 | 0.7066 | 0.8875 | 0.8203 | 0.8255 | 0.8393 | 0.9031 | 10.3222 |
| A6 positive_only | 13 | 0.6887 | 0.8706 | 0.8062 | 0.8122 | 0.8283 | 0.8939 | 7.9111 |
| A6 positive_only | 37 | 0.6906 | 0.8738 | 0.8079 | 0.8100 | 0.8330 | 0.8912 | 10.9222 |
| A6 positive_only | 73 | 0.6930 | 0.8754 | 0.8094 | 0.8127 | 0.8335 | 0.9025 | 8.1667 |

## Deltas appariés (bootstrap 10 000  IC95 par percentiles)

| Comparaison | Métrique | Delta moyen | IC95 | gains/pertes/nuls | Verdict |
|:--|:--|--:|:--:|:--:|:--|
| A2-A1 | iou | -0.0032 | [-0.0054 ; -0.0009] | 102/168/0 | défavorable |
| A2-A1 | iou_buffered_tol3 | +0.0020 | [-0.0004 ; +0.0045] | 124/146/0 | indiscernable |
| A2-A1 | dice | -0.0014 | [-0.0031 ; +0.0003] | 102/168/0 | indiscernable |
| A2-A3 | iou | -0.0000 | [-0.0022 ; +0.0021] | 139/131/0 | indiscernable |
| A2-A3 | iou_buffered_tol3 | +0.0020 | [-0.0004 ; +0.0044] | 138/132/0 | indiscernable |
| A2-A3 | dice | +0.0009 | [-0.0008 ; +0.0027] | 139/131/0 | indiscernable |
| A2-A4 | iou | +0.0015 | [-0.0004 ; +0.0035] | 151/119/0 | indiscernable |
| A2-A4 | iou_buffered_tol3 | +0.0015 | [-0.0005 ; +0.0036] | 143/127/0 | indiscernable |
| A2-A4 | dice | +0.0016 | [+0.0001 ; +0.0031] | 151/119/0 | favorable |
| A2-A5 | iou | +0.0013 | [-0.0007 ; +0.0037] | 136/134/0 | indiscernable |
| A2-A5 | iou_buffered_tol3 | +0.0013 | [-0.0006 ; +0.0035] | 151/119/0 | indiscernable |
| A2-A5 | dice | +0.0007 | [-0.0007 ; +0.0023] | 136/134/0 | indiscernable |
| A2-A6 | iou | +0.0156 | [+0.0119 ; +0.0195] | 193/77/0 | favorable |
| A2-A6 | iou_buffered_tol3 | +0.0143 | [+0.0102 ; +0.0188] | 170/100/0 | favorable |
| A2-A6 | dice | +0.0120 | [+0.0091 ; +0.0151] | 193/77/0 | favorable |
| A2-A0 | iou | +0.0391 | [+0.0320 ; +0.0467] | 214/56/0 | favorable |
| A2-A0 | iou_buffered_tol3 | +0.0472 | [+0.0380 ; +0.0571] | 189/81/0 | favorable |
| A2-A0 | dice | +0.0312 | [+0.0248 ; +0.0383] | 214/56/0 | favorable |
| A1-A0 | iou | +0.0422 | [+0.0349 ; +0.0502] | 223/47/0 | favorable |
| A1-A0 | iou_buffered_tol3 | +0.0451 | [+0.0360 ; +0.0550] | 192/78/0 | favorable |
| A1-A0 | dice | +0.0326 | [+0.0260 ; +0.0398] | 223/47/0 | favorable |
| A4-A1 | iou | -0.0047 | [-0.0065 ; -0.0029] | 91/179/0 | défavorable |
| A4-A1 | iou_buffered_tol3 | +0.0005 | [-0.0013 ; +0.0023] | 119/151/0 | indiscernable |
| A4-A1 | dice | -0.0030 | [-0.0044 ; -0.0017] | 91/179/0 | défavorable |
