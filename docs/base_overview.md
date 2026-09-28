# Project Overview

## Cíl

Projekt automatizuje celý řetězec od parametrického NACA profilu po vyhodnocený ML model:

- generování sampling table a blockMesh cases;
- steady incompressible RANS výpočty v OpenFOAMu;
- archivaci CFD výsledků a převod polí do pravidelné mřížky;
- kontrolu kvality dat;
- reprodukovatelné experimenty s pevným test setem;
- samostatnou evaluaci spatial a Cl/Cd modelů.

## Aktivní komponenty

| Vrstva | Implementace |
|---|---|
| Sampling | `src/generate_sampling_table.py`, `src/sampling.py` |
| Case build | `src/case_builder.py`, `src/blockmesh_generator.py`, `src/file_editors.py` |
| CFD běh a archiv | `src/case_runner.py` |
| Extrakce polí | `src/flow_extractor.py` |
| QA datasetu | `scripts/inspect_flow_npz.py`, `src/inspect_plausibility.py` |
| Dataset | `src/ml_dataset.py`, `src/ml_clcd_dataset.py` |
| Modely | `src/ml_models.py` |
| Splity a experimenty | `src/ml_data_split.py`, `src/ml_experiment.py`, `src/ml_hyperparameter_search.py` |
| Device | `src/ml_device.py` |
| Evaluace | `src/evaluate_*`, `src/clcd_inference.py` |

`src/BACKUPblockmesh_generator.py` je pouze referenční záloha a aktivní workflow jej nepoužívá. STL/`snappyHexMesh` pipeline je také neaktivní.

## Datový životní cyklus

1. `build_blockmesh_cases.py` připraví pracovní OpenFOAM cases.
2. `run_cases.py` provede CFD i `foamToVTK`.
3. Runner archivuje potřebné výstupy do `flow_fields_output/<case_id>` a úspěšný pracovní case smaže.
4. `extract_flow_fields.py` přidá do téhož case adresáře pravidelně vzorkované NPZ.
5. `inspect_flow_npz.py` vytvoří QA výsledky, které datasety používají jako exclusion list.
6. Trénovací skript vytvoří perzistentní test split, provede výběr konfigurace pouze na development datech a uloží finální model s test metrikami.

## Vstupy modelů

Spatial modely dostávají tensor s pěti kanály a predikují tři kanály `[p, Ux, Uy]`. Dataset sestavuje vstupy z masky/geometrie a provozních podmínek uložených v NPZ a názvu case.

`clcd_mlp` dostává:

```text
[camber_percent, camber_position_tenths, thickness_percent, aoa_deg, inlet_velocity]
```

a predikuje `[Cl, Cd]`. Targety jsou průměry z posledních `tail_window` validních řádků `forceCoeffs.dat`; dataset požaduje nejméně 100 finite řádků.

## Zdroj pravdy

Chování určují primárně skripty a následující konfigurace:

- [paths.yaml](../configs/paths.yaml)
- [dataset_config.yaml](../configs/dataset_config.yaml)
- [solver_config.yaml](../configs/solver_config.yaml)
- [ml_models_config.yaml](../configs/ml_models_config.yaml)

Některé provozní konstanty jsou stále přímo ve skriptech: extrakční mřížka a počet extraction workers, stejně jako limity detekce divergence `simpleFoam`.
