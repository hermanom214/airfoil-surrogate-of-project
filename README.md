# Airfoil Surrogate Model - OpenFOAM + ML Pipeline

## O projektu

Projekt implementuje end-to-end pipeline pro surrogate modelovani 2D obtokoveho CFD nad NACA profily.

Aktualne existuji dve paralelni ML vetve:

1. Pole p, Ux, Uy z regularni mrizky (simple_unet, rans_pinn)
2. Skalarni koeficienty Cl, Cd z OpenFOAM forceCoeffs (clcd_mlp)

Obe vetve sdileji stejnou CFD pripravu pripadu, lisí se jen datovou reprezentaci, modelem a evaluaci.

## Hlavni workflow

### 1. Sestaveni OpenFOAM pripadu

```bash
python -m scripts.build_blockmesh_cases
```

- Vygeneruje nebo nacte sampling table.
- Vytvori case adresare z OpenFOAM sablony.
- Vygeneruje blockMeshDict a upravi 0/U a system/forceCoeffs.

### 2. Spusteni CFD

```bash
python -m scripts.run_cases
```

- Spousti retezec blockMesh -> checkMesh -> decomposePar -> simpleFoam -parallel -> reconstructPar.
- Uklada logy a run_status.csv.

### 3. Extrakce flow fields (pro vetev p/U)

```bash
python -m scripts.extract_flow_fields
```

- Spousti foamToVTK.
- Interpoluje p a U na regularni mrizku.
- Uklada NPZ dataset pro spatial modely.

### 4. Trenovani modelu

```bash
python -m scripts.train_ml_model
```

Podle model.name v configs/ml_models_config.yaml:

- simple_unet nebo rans_pinn: trenink nad NPZ flow-fields.
- clcd_mlp: trenink nad scalar features -> [Cl, Cd] z forceCoeffs.dat.

U clcd_mlp je workflow robustni proti nekvalitnim CFD vystupum:

- parsuji se pouze finite hodnoty (Time, Cd, Cl),
- vyzaduje se minimalni pocet validnich iteraci,
- nevalidni pripady se skipuji se zaznamenanym duvodem,
- normalizace i loss jsou kontrolovany proti NaN/Inf,
- checkpoint se neuklada pri nevalidni historii loss.

### 5. Evaluace modelu

#### Evaluace spatial modelu (beze zmen)

```bash
python -m scripts.evaluate_ml_model
```

- Pro simple_unet a rans_pinn.
- Per-case a global metriky pro pole p/U.
- Pole a velocity diagnosticke obrazky.

#### Evaluace scalar Cl/Cd modelu (nova paralelni vetev)

```bash
python -m scripts.evaluate_clcd_model
```

- Pouziva checkpoint normalizaci (nerekonstruuje statistiky).
- Preferuje val_case_ids z checkpointu, fallback na val_indices, az pak rebuild splitu.
- Uklada:
	- data/models/clcd_mlp_validation/metrics/validation_metrics_per_case.csv
	- data/models/clcd_mlp_validation/metrics/validation_summary.json
	- data/models/clcd_mlp_validation/figures/cl_scatter.png
	- data/models/clcd_mlp_validation/figures/cd_scatter.png
	- data/models/clcd_mlp_validation/figures/cl_residual.png
	- data/models/clcd_mlp_validation/figures/cd_residual.png
	- data/models/clcd_mlp_validation/figures/cl_errors.png
	- data/models/clcd_mlp_validation/figures/cd_errors.png

## Modely

- simple_unet: baseline surrogate pro p/U pole.
- rans_pinn: CNN s fyzikalnim residual loss.
- clcd_mlp: MLP pro regresi [Cl, Cd] z feature vectoru:
	[camber_percent, camber_position_tenths, thickness_percent, aoa_deg, inlet_velocity].

Checkpoint clcd_mlp uklada model_state_dict + normalizaci + poradi feature/target + train/val split metadata.

## Pomocne skripty

| Skript | Popis |
|---|---|
| scripts/inspect_flow_npz.py | Kontrola obsahu NPZ souboru |
| scripts/plot_residuals.py | Vykresleni OpenFOAM residuals |
| scripts/evaluate_ml_model.py | Evaluace spatial modelu (p/U) |
| scripts/evaluate_clcd_model.py | Evaluace scalar Cl/Cd modelu |

## Konfigurace

- configs/paths.yaml
- configs/dataset_config.yaml
- configs/solver_config.yaml
- configs/ml_models_config.yaml

## Dokumentace

Detailni technicky popis je v docs:

- docs/base_overview.md
- docs/pipeline.md
- docs/cfd.md
- docs/mesh.md
- docs/ml_model.md
- docs/project_plan.md