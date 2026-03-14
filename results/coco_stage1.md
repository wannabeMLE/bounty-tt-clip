# COCO Zero-Shot Validation — Stage 1

**Images:** 20 | **Classes:** 80 | **Match rate:** 18/20 (90%) | **Avg PCC:** 0.923920

| # | PyTorch Top-1 | TTNN Top-1 | Match | PCC |
|---|---------------|------------|-------|-----|
| 1 | dining table (67%) | dining table (33%) | YES | 0.8741 |
| 2 | bear (91%) | bear (94%) | YES | 0.9375 |
| 3 | bed (86%) | bed (50%) | YES | 0.9166 |
| 4 | stop sign (100%) | stop sign (100%) | YES | 0.9646 |
| 5 | teddy bear (97%) | teddy bear (94%) | YES | 0.9190 |
| 6 | skis (80%) | skis (64%) | YES | 0.9482 |
| 7 | refrigerator (90%) | refrigerator (87%) | YES | 0.9399 |
| 8 | sports ball (50%) | sports ball (53%) | YES | 0.9340 |
| 9 | sports ball (70%) | sports ball (66%) | YES | 0.9097 |
| 10 | tennis racket (84%) | tennis racket (69%) | YES | 0.8889 |
| 11 | bench (24%) | bench (29%) | YES | 0.9054 |
| 12 | cell phone (72%) | cell phone (90%) | YES | 0.9013 |
| 13 | fire hydrant (28%) | fire hydrant (27%) | YES | 0.8835 |
| 14 | sandwich (44%) | cake (42%) | NO | 0.9439 |
| 15 | kite (53%) | boat (19%) | NO | 0.9560 |
| 16 | laptop (70%) | laptop (87%) | YES | 0.9427 |
| 17 | traffic light (50%) | traffic light (47%) | YES | 0.9171 |
| 18 | bus (99%) | bus (99%) | YES | 0.9115 |
| 19 | laptop (59%) | laptop (79%) | YES | 0.9410 |
| 20 | airplane (88%) | airplane (82%) | YES | 0.9434 |

**Mismatches:** 2
- Image 14: PT=sandwich vs TT=cake
- Image 15: PT=kite vs TT=boat
