# ML Models and Reproducible Training

## Modely

| Model | Vstup | Výstup | Účel |
|---|---|---|---|
| `simple_unet` | `(B, 5, H, W)` | `(B, 3, H, W)` | baseline pro `[p, Ux, Uy]` |
| `rans_pinn` | `(B, 5, H, W)` | `(B, 3, H, W)` | convolutional model s data + PDE residual loss |
| `clcd_mlp` | `(B, 5)` | `(B, 2)` | regrese `[Cl, Cd]` |

Architektury jsou v [src/ml_models.py](../src/ml_models.py). Spatial output channel order je vždy `[p, Ux, Uy]`.

`simple_unet` je encoder-decoder se skip connections. `rans_pinn` je fully convolutional síť a během tréninku přidává maskované continuity/momentum residuals po nakonfigurovaném warmupu. `clcd_mlp` je plně propojená síť s konfigurovatelnými hidden layers a dropoutem.

## Datasety a filtrování

[src/ml_dataset.py](../src/ml_dataset.py) hledá `flow_fields_output/*/*.npz`. [src/ml_clcd_dataset.py](../src/ml_clcd_dataset.py) hledá `flow_fields_output/case_*` a v nich nakonfigurovaný `forceCoeffs.dat`.

Scalar feature order:

```text
camber_percent, camber_position_tenths, thickness_percent, aoa_deg, inlet_velocity
```

Scalar target order je `Cl, Cd`. Parser vynechává nečíselné a non-finite řádky, požaduje nejméně 100 validních řádků a zprůměruje posledních `tail_window` řádků (aktuálně 200). Oba datasety vynechávají cases označené jako nevyhovující v inspection výsledcích.

## Experiment protocol

[scripts/train_ml_model.py](../scripts/train_ml_model.py) používá stejný protokol pro všechny modely:

1. načte dataset a jeho stabilní case ID;
2. vytvoří nebo načte `fixed_test_split.json`;
3. test cases úplně vyřadí z výběru modelu;
4. na development části provede holdout nebo CV;
5. normalizaci fituje zvlášť pouze na training indices každého foldu;
6. vybere nejlepší hyperparametry podle validačního objective;
7. finální model trénuje na celé development části s nově fitovanou normalizací;
8. spočítá finální metriky na fixed test setu a uloží checkpoint.

Strategie `none` vypíná hledání hyperparametrů, nikoli automaticky CV. CV řídí model-specific `experiments.<model>.data_split.cv_enabled` nebo CLI přepínače. `--single-run` explicitně vypíná CV a vyžaduje search strategy `none`.

Podporované strategie:

- split/CV: `kfold`, `group_kfold_naca`;
- search: `none`, `randomized`, `manual_coarse`.

Pokud se změní dataset fingerprint, existující fixed split se automaticky nepřepíše. Novou test populaci lze vytvořit pouze pomocí `--regenerate-split` nebo odpovídajícího config flagu.

## CLI

```powershell
python -m scripts.train_ml_model --model simple_unet
python -m scripts.train_ml_model --model rans_pinn --device cuda
python -m scripts.train_ml_model --model clcd_mlp --device cpu
python -m scripts.train_ml_model --model simple_unet --cv-strategy group_kfold_naca
python -m scripts.train_ml_model --model simple_unet --cross-validation
python -m scripts.train_ml_model --model rans_pinn --search-strategy none --single-run
python -m scripts.train_ml_model --model simple_unet --epochs 3 --smoke-test
python -m scripts.train_ml_model --model simple_unet --regenerate-split
```

`--device auto|cpu|cuda` přepisuje config. Explicitní `cuda` skončí chybou, pokud CUDA není dostupná. `--smoke-test` omezuje běh na 2 epochy, nejvýše 2 kandidáty a 2 folds; stále ale projde stejným split/training protokolem.

## Artefakty

Produkční běh zapisuje do `data/models/<model_name>/`, smoke běh do `data/models/<model_name>/smoke_test/`:

- `<model_name>_airfoil.pt`;
- `split_metadata.json`;
- `cv_results.json`;
- `hyperparameter_search_results.csv`;
- `final_test_metrics.json`.

Perzistentní `fixed_test_split.json` zůstává v kořeni modelu i pro smoke běh. Checkpoint obsahuje model state/config, normalizační statistiky, dataset fingerprint, development/test case IDs, nejlepší hyperparametry, experiment mode, seed/device a runtime verze.

## Evaluace

```powershell
python -m scripts.evaluate_ml_model --device auto
python -m scripts.evaluate_clcd_model --device auto
```

Spatial evaluace načte model vybraný v `model.name`; je určena jen pro `simple_unet` a `rans_pinn`. Výstupy ukládá do `data/models/<model_name>_validation/` jako per-case CSV, summary JSON, dataset error plot a per-case field/velocity obrázky.

Cl/Cd evaluace vždy používá `clcd_mlp` checkpoint a zapisuje per-case CSV, summary JSON a Cl/Cd scatter, residual a sorted-error grafy do `data/models/clcd_mlp_validation/`.

U nových checkpointů označuje název adresáře historicky „validation“, ale skutečný `evaluation_split` je fixed test set. Staré checkpointy bez `test_case_ids` používají uloženou nebo znovu sestavenou legacy validation část.
