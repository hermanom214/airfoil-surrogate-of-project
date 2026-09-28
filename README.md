# Airfoil Surrogate Model – OpenFOAM + ML

Reprodukovatelná pipeline pro generování 2D NACA profilů, výpočty v OpenFOAMu a trénování surrogate modelů.

Aktivní jsou dvě ML větve:

- `simple_unet` a `rans_pinn` predikují pole `[p, Ux, Uy]` na pravidelné mřížce;
- `clcd_mlp` predikuje skalární koeficienty `[Cl, Cd]` z parametrů profilu a proudění.

Aktivní meshing používá pouze procedurálně generovaný `blockMeshDict`. Starší STL/`snappyHexMesh` větev není součástí workflow.

## Požadavky a konfigurace

Python závislosti jsou v `requirements.txt`; doplňky pro notebooky v `requirements-jupyter.txt`. OpenFOAM se na Windows spouští přes Cygwin bash nastavený v [configs/paths.yaml](configs/paths.yaml).

Před spuštěním zkontrolujte absolutní cesty v:

- [configs/paths.yaml](configs/paths.yaml) – projekt, OpenFOAM, sampling table a datový archiv;
- [configs/dataset_config.yaml](configs/dataset_config.yaml) – sampling, šablona a parametry sítě;
- [configs/solver_config.yaml](configs/solver_config.yaml) – CFD příkazy, MPI a paralelismus cases;
- [configs/ml_models_config.yaml](configs/ml_models_config.yaml) – model, device, split, CV, search a výstupy.

## Aktuální workflow

### 1. Sestavení cases

```powershell
python -m scripts.build_blockmesh_cases
```

Skript vždy znovu vytvoří sampling table, sestaví cases ze šablony, zapíše `blockMeshDict`, upraví `0/U` a `system/forceCoeffs`, uloží `params.json` a zkopíruje cases do OpenFOAM simulačního adresáře.

> Pozor: před buildem smaže všechny podadresáře v obou spravovaných kořenech `blockmesh_cases` (run root i simulation mirror).

### 2. CFD, VTK a archivace

```powershell
python -m scripts.run_cases
python -m scripts.run_cases --start-case-id 30
```

Pro každý case proběhne:

`blockMesh → checkMesh → decomposePar -force → simpleFoam -parallel → reconstructPar -latestTime → foamToVTK`

Úspěšný case se archivuje do `flow_fields_output/<case_id>` z `configs/paths.yaml`. Archiv obsahuje `logs`, `postProcessing`, `system`, `VTK`, poslední časový adresář a případně `params.json`. Poté je pracovní case ze simulačního adresáře odstraněn. Selhaný case zůstává na místě k diagnostice. Souhrn je v `run_status.csv`.

### 3. Extrakce polí do NPZ

```powershell
python -m scripts.extract_flow_fields
```

Skript čte archivované case adresáře přímo z `flow_fields_output`, načte již vytvořené VTK, provede kontrolu plausibility a interpoluje `p` a `U` na mřížku `640 × 320` v oblasti `x ∈ [-0.75, 1.75]`, `y ∈ [-0.75, 0.75]`. Výsledek uloží jako `<case_id>/<case_id>.npz` a aktualizuje `flow_dataset_index.csv`.

### 4. Kontrola datasetu

```powershell
python -m scripts.inspect_flow_npz
```

Kontrola vytváří souhrn, diagnostické obrázky a `inspect_plausibility.csv`. Cases označené jako nevyhovující jsou následně vynechány oběma datasety.

### 5. Trénování

```powershell
python -m scripts.train_ml_model --model simple_unet
python -m scripts.train_ml_model --model rans_pinn
python -m scripts.train_ml_model --model clcd_mlp
```

Model lze vybrat argumentem `--model`; bez něj se použije `model.name` z konfigurace. Každý běh rezervuje perzistentní fixed test set. Výběr modelu probíhá na development části pomocí holdoutu nebo CV a volitelného hledání hyperparametrů. Normalizace se fituje pouze z trénovacích dat daného foldu a pro finální model ze všech development dat.

Užitečné přepínače:

```powershell
python -m scripts.train_ml_model --model simple_unet --smoke-test
python -m scripts.train_ml_model --model simple_unet --device cpu --epochs 3
python -m scripts.train_ml_model --model simple_unet --cross-validation
python -m scripts.train_ml_model --model rans_pinn --search-strategy none --single-run
```

Artefakty se ukládají do `data/models/<model_name>/`: checkpoint, split metadata, CV/search výsledky a finální test metriky. Smoke test používá podadresář `smoke_test`.

### 6. Evaluace uloženého checkpointu

```powershell
python -m scripts.evaluate_ml_model --device auto
python -m scripts.evaluate_clcd_model --device auto
```

První skript je pro spatial model nastavený v konfiguraci, druhý pro `clcd_mlp`. Nové checkpointy se vyhodnocují na uloženém fixed test setu; fallback na validační split slouží jen pro starší checkpointy.

## Dokumentace

- [Přehled projektu](docs/base_overview.md)
- [End-to-end pipeline](docs/pipeline.md)
- [CFD workflow](docs/cfd.md)
- [Mesh workflow](docs/mesh.md)
- [ML modely a experimenty](docs/ml_model.md)
- [Aktuální stav a další kroky](docs/project_plan.md)
- [Historický CPU/GPU benchmark](docs/benchmark_cpu_gpu_cuda.md)
