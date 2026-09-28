# Mesh Workflow (blockMesh-only)

## Generování

Aktivní síť je procedurální C-grid kolem NACA 4-digit profilu. Nepoužívá STL ani `snappyHexMesh`.

Pro každý řádek sampling table:

1. [src/case_builder.py](../src/case_builder.py) zkopíruje `templates/openfoam_base_case_yPlus1`;
2. [src/blockmesh_generator.py](../src/blockmesh_generator.py) vytvoří horní a dolní křivku profilu s LE clusteringem;
3. profil se natočí o case AoA kolem čtvrtiny tětivy;
4. z konfigurace vzniknou body, bloky, spline edges a boundaries C-gridu;
5. zapíše se `system/blockMeshDict`, `0/U`, `system/forceCoeffs` a `params.json`.

Vstupní rychlost mění `0/U` a referenční `magUInf` ve `forceCoeffs`, nikoli topologii sítě.

## Aktuální sampling

Hodnoty z [configs/dataset_config.yaml](../configs/dataset_config.yaml):

- camber: `0, 1, 2` %;
- pozice camberu: `2, 4` desetiny tětivy; pro nulový camber se tvoří symetrický `00xx` profil;
- tloušťka: `8, 10, 12, 14, 16` %;
- AoA: `-4, -2, 0, 2, 4` stupňů;
- inlet velocity: `15, 17.5, 20, 22.5, 25 m/s`.

Sampling table se při každém spuštění build skriptu regeneruje z těchto hodnot.

## Aktuální parametry sítě

| Skupina | Hodnoty |
|---|---|
| Doména | `x_min=-5`, `x_max=12`, `x_far=20`, `y_min=-5`, `y_max=5`, `z_half=0.05` |
| Rozlišení | `n_airfoil_half=520`, `n_streamwise_near=240`, `n_wall_normal=170`, `n_wake_x=320`, `n_far_wake_x=80`, `n_z=1` |
| Grading | `grading_to_wall=2400`, `grading_le_tangent=0.65`, `grading_wake_x=1.35`, `grading_far_wake_x=4.0` |
| LE | `le_cluster_exp=2.8`, `enable_le_cap=false`, `le_topology_fraction=0.028`, `n_le_cap_normal=42` |
| TE/wake | `te_transition_fraction=0.995`, `wake_cut_length=0.0001` |

`frontAndBack` používá `empty`, takže jedna buňka přes tloušťku představuje 2D úlohu.

## Build a validace

```powershell
python -m scripts.build_blockmesh_cases
python -m scripts.run_cases
```

Build skript před vytvořením nové sady odstraní všechny podadresáře v obou spravovaných `blockmesh_cases` kořenech. Runner následně pro každý case spustí `blockMesh` a `checkMesh` před solverem. Pokud kterýkoli krok selže, další CFD kroky daného case se nespustí.

Každá změna parametrů sítě by měla být nejdříve ověřena na malé reprezentativní sadě profilů, AoA a rychlostí; skript zatím nemá subset/smoke přepínač pro build ani CFD.
