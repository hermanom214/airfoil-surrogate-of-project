# CFD Workflow (OpenFOAM)

## Solver chain

Aktuální řetězec jednoho case je:

1. `blockMesh`
2. `checkMesh`
3. `decomposePar -force`
4. `simpleFoam -parallel`
5. `reconstructPar -latestTime`
6. `foamToVTK -latestTime -ascii -fields '(p U)'`
7. archivace vybraných výsledků
8. odstranění úspěšného pracovního case

Prvních pět kroků a `foamToVTK` jsou v [configs/solver_config.yaml](../configs/solver_config.yaml). Implementace, divergence guard, archivace a cleanup jsou v [src/case_runner.py](../src/case_runner.py).

## Spuštění

```powershell
python -m scripts.run_cases
python -m scripts.run_cases --start-case-id 30
```

Runner používá `run_cases.max_workers` pro počet současných cases a `case_runner.n_procs` pro `mpiexec -np` uvnitř jednoho case. Adresáře bez `0`, `constant` a `system` ignoruje; filtr `--start-case-id` funguje jen pro názvy `case_<id>_...`.

## Fyzikální a numerické nastavení šablony

Šablona [openfoam_base_case_yPlus1](../templates/openfoam_base_case_yPlus1) používá:

- steady incompressible RANS solver `simpleFoam`;
- model `kOmegaSST`;
- kinematickou viskozitu `1.5e-5 m²/s`;
- low-Re wall treatment se sítí cílenou přibližně na `y+ ≈ 1`;
- `empty` front/back boundary pro 2D výpočet.

Rychlostní inlet je pro každý case přepsán builderem. `forceCoeffs` dostává odpovídající `magUInf`, takže jeho historie slouží jako target scalar ML větve.

`controlDict` provádí nejvýše 1000 SIMPLE iterací (`endTime=1000`) a zapisuje koncový stav (`writeInterval=1000`, `purgeWrite=1`, binary + compression). Residual control může solver ukončit dříve.

Hlavní schémata jsou `steadyState`, `Gauss linear`, bounded `linearUpwind` pro `U`, limited linear pro turbulence scalars a corrected laplacian/snGrad. Přesné aktuální hodnoty je nutné číst přímo z `system/fvSchemes` a `system/fvSolution` šablony.

## Detekce divergence

Během `simpleFoam` runner sleduje log. Proces ukončí, pokud řádek lineárního solveru současně hlásí 1000 iterací a:

- final residual je NaN nebo Inf; nebo
- absolutní final residual je alespoň `1e2`.

Takový case dostane stav `nOK(...)` a zbývající kroky se přeskočí.

## Logy, archiv a cleanup

Per-case logy:

```text
logs/01_blockMesh.log
logs/02_checkMesh.log
logs/03_decomposePar.log
logs/04_simpleFoam.log
logs/05_reconstructPar.log
logs/06_foamToVTK.log
```

Po úspěchu se do `flow_fields_output/<case_id>` kopírují `logs`, `postProcessing`, `system`, `VTK`, nejnovější numerický time adresář a případný `params.json`.

Archivace záměrně nepřepisuje existující case adresář. Pokud cíl existuje, status je `target_exists(...)` a pracovní case zůstane. Po úspěšné archivaci se pracovní simulation case odstraní. Při selhání během solver chain se odstraní `processor*`; při selhání `foamToVTK` nebo archivace se case ponechá pro diagnostiku.

`run_status.csv` obsahuje stavy blockMesh, checkMesh, simpleFoam, foamToVTK, archivace, cleanup, celkový status a runtime.
