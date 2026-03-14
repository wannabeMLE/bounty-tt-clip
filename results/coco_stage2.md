# COCO Zero-Shot Validation — Stage 2

**Images:** 20 | **Classes:** 80 | **Match rate:** 18/20 (90%) | **Avg PCC:** 0.926669

| # | PyTorch Top-1 | TTNN Top-1 | Match | PCC |
|---|---------------|------------|-------|-----|
| 1 | dining table (67%) | dining table (38%) | YES | 0.8806 |
| 2 | bear (91%) | bear (95%) | YES | 0.9404 |
| 3 | bed (86%) | bed (48%) | YES | 0.9253 |
| 4 | stop sign (100%) | stop sign (100%) | YES | 0.9677 |
| 5 | teddy bear (97%) | teddy bear (95%) | YES | 0.9265 |
| 6 | skis (80%) | skis (74%) | YES | 0.9473 |
| 7 | refrigerator (90%) | refrigerator (90%) | YES | 0.9346 |
| 8 | sports ball (50%) | sports ball (66%) | YES | 0.9333 |
| 9 | sports ball (70%) | sports ball (78%) | YES | 0.9154 |
| 10 | tennis racket (84%) | tennis racket (63%) | YES | 0.9010 |
| 11 | bench (24%) | bench (27%) | YES | 0.9035 |
| 12 | cell phone (72%) | cell phone (91%) | YES | 0.9136 |
| 13 | fire hydrant (28%) | fire hydrant (29%) | YES | 0.8797 |
| 14 | sandwich (44%) | cake (50%) | NO | 0.9444 |
| 15 | kite (53%) | surfboard (29%) | NO | 0.9477 |
| 16 | laptop (70%) | laptop (86%) | YES | 0.9438 |
| 17 | traffic light (50%) | traffic light (45%) | YES | 0.9174 |
| 18 | bus (99%) | bus (99%) | YES | 0.9188 |
| 19 | laptop (59%) | laptop (83%) | YES | 0.9479 |
| 20 | airplane (88%) | airplane (72%) | YES | 0.9444 |

**Mismatches:** 2
- Image 14: PT=sandwich vs TT=cake
- Image 15: PT=kite vs TT=surfboard
