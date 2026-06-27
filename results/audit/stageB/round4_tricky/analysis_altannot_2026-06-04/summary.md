# Round-4 Alternative-Annotator Robustness Test (Plan B)

- annotators present: **['Yang', 'Yu', 'Alexander', 'Heejin']**
- annotators missing: **['Jingwei', 'Tobi']**
- total labels loaded: 112  | bootstrap B=2000 | infer_sc=False

> Caveats: label-level (not majority) rates; CIs cluster-bootstrapped by row_id; alpha is nominal Krippendorff. With <6 annotators or thin overlap, LOAO is a 'does one rater drive it' probe, NOT full generalizability. This is the worst-case (tricky score-3/7) subset; fragility here does not condemn the easy bins.

## Scope: `round4_new`  (labels=84)

- co-labeled rows (>=2 labels): **39**; labels-per-row dist: 1->3, 2->36, 3->3
- pairwise overlap: Alexander&Heejin=4, Alexander&Yang=11, Alexander&Yu=9, Heejin&Yang=8, Heejin&Yu=10, Yang&Yu=3
- per-annotator strict-pass: Yang=0.333, Yu=0.6, Alexander=0.529, Heejin=0.278
- **strict pass**: score3=0.297 0.162-0.432 | score7=0.583 0.432-0.73 | **gap=0.286 0.079-0.486**
- **LOAO**: delta range [0.09, 0.381], sign-stable=True; drops that move delta >50%: ['-Yu']
- agreement (full): cwf alpha=0.662/pair=0.933(n=39), ent alpha=0.171/ord=0.202/pair=0.537(n=37), sc alpha=-0.003/ord=-0.058/pair=0.419(n=37), ps alpha=-0.023/pair=0.5(n=32)
- **Heejin check**: strict_rate=0.278; cwf: Heejin=0.909(n=22) others=0.957(n=23), ent: Heejin=0.524(n=21) others=0.55(n=20), sc: Heejin=0.364(n=22) others=0.476(n=21), ps: Heejin=0.5(n=18) others=0.5(n=18)

## Scope: `all37`  (labels=112)

- co-labeled rows (>=2 labels): **44**; labels-per-row dist: 1->21, 2->41, 3->3
- pairwise overlap: Alexander&Heejin=4, Alexander&Yang=13, Alexander&Yu=10, Heejin&Yang=8, Heejin&Yu=10, Yang&Yu=5
- per-annotator strict-pass: Yang=0.407, Yu=0.679, Alexander=0.444, Heejin=0.278
- **strict pass**: score3=0.304 0.178-0.444 | score7=0.611 0.48-0.741 | **gap=0.307 0.116-0.483**
- **LOAO**: delta range [0.18, 0.373], sign-stable=True; drops that move delta >50%: none
- agreement (full): cwf alpha=0.667/pair=0.94(n=44), ent alpha=0.213/ord=0.267/pair=0.565(n=42), sc alpha=0.071/ord=0.049/pair=0.458(n=42), ps alpha=0.115/pair=0.561(n=37)
- **Heejin check**: strict_rate=0.278; cwf: Heejin=0.909(n=22) others=0.964(n=28), ent: Heejin=0.524(n=21) others=0.6(n=25), sc: Heejin=0.364(n=22) others=0.538(n=26), ps: Heejin=0.5(n=18) others=0.609(n=23)
