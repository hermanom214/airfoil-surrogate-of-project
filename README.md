# Airfoil Surrogate Model – OpenFOAM + ML Pipeline

## O projektu

Projekt buduje end-to-end pipeline pro **náhradu (surrogate) CFD simulací** obtékání profilů NACA pomocí hlubokého učení. Cílem je natrénovat neuronovou síť, která na základě geometrie profilu a podmínek proudění (úhel náběhu, rychlost) předpoví celá 2D pole tlaku a rychlosti – bez nutnosti spouštět OpenFOAM pro každý nový případ.

### Cíl

1. Parametricky vygenerovat sadu NACA profilů a kombinovat je s různými úhly náběhu a vstupními rychlostmi.
2. Pro každou kombinaci postavit OpenFOAM případ s C-grid sítí (`blockMesh`).
3. Spustit CFD simulace (`simpleFoam`) a extrahovat výsledná pole **p**, **U**.
4. Interpolovat pole na pravidelnou mřížku a uložit je jako `.npz` dataset.
5. Natrénovat **U-Net surrogate model** (případně s fyzikálními ztrátami) na tomto datasetu.

---

## Workflow – krok za krokem

Pipeline se skládá ze čtyř hlavních kroků. Každý má vlastní spouštěcí skript.

### 1. Sestavení OpenFOAM případů

```bash
python -m scripts.build_blockmesh_cases
```

- Pokud neexistuje, vygeneruje vzorkovací tabulku (`geometry/generated_profiles/airfoil_sampling_table.csv`) se všemi kombinacemi NACA profilů × AoA × rychlostí.
- Pro každý řádek tabulky vytvoří adresář případu z šablony `templates/openfoam_base_case_yPlus1/`.
- Vygeneruje case-specifický `blockMeshDict` (C-grid topologie) a aktualizuje `0/U` a `system/forceCoeffs`.
- Výstup: adresáře případů v `run/airfoil_surrogate_cases/blockmesh_cases/`.

**Klíčová konfigurace:** `configs/dataset_config.yaml`, `configs/paths.yaml`

---

### 2. Spuštění CFD simulací

> Spouštět v terminálu OpenFOAM (Cygwin/WSL), z kořene projektu:

```bash
python -m scripts.run_cases
```

- Spustí celý CFD řetězec pro každý případ: `blockMesh` → `checkMesh` → `decomposePar` → `simpleFoam -parallel` → `reconstructPar`.
- Případy běží paralelně (konfigurovatelný počet workers).
- Výsledky a statusy zapisuje do `run_status.csv`.
- Po skončení maže dočasné adresáře `processor*/`.

**Klíčová konfigurace:** `configs/solver_config.yaml`, `configs/paths.yaml`

---

### 3. Extrakce proudových polí

```bash
python -m scripts.extract_flow_fields
```

- Pro každý dokončený případ spustí `foamToVTK` a načte výsledná VTK data.
- Interpoluje pole **p** a **U** na pravidelnou mřížku 640 × 320 bodů (rozsah x ∈ [−0.75, 1.75], y ∈ [−0.75, 0.75]).
- Vytvoří masku profilu (body uvnitř tělesa → `fluid_mask = 0`).
- Uloží komprimovaný `.npz` soubor na případ + souhrnný index `flow_dataset_index.csv`.

**Klíčová konfigurace:** `configs/paths.yaml`

---

### 4. Trénování surrogate modelu

```bash
python -m scripts.train_ml_model
```

- Načte `.npz` dataset, rozdělí na trénovací a validační sadu.
- Trénuje **SimpleUNet** (nebo fyzikálně informovanou CNN) – model predikuje 2D pole p, Ux, Uy z podmínek proudění.
- Ztráta je masked MSE počítaná pouze nad tekutinovou doménou (mimo těleso profilu).
- Uloží váhy modelu a vykresli křivky trénování.

**Klíčová konfigurace:** `configs/ml_models_config.yaml`, `configs/paths.yaml`

---

## Pomocné skripty

| Skript | Popis |
|---|---|
| `scripts/inspect_flow_npz.py` | Zobrazí obsah a statistiky jednoho `.npz` souboru s extrahovanými poli |
| `scripts/plot_residuals.py` | Vykreslí konvergenční residuály z logů OpenFOAM simulace |

---

## Struktura projektu

```
configs/            # YAML konfigurace (cesty, sampling, solver, ML)
docs/               # Detailní dokumentace jednotlivých částí
geometry/           # Vygenerovaná vzorkovací tabulka profilů
run/                # Adresáře OpenFOAM případů
scripts/            # Hlavní spouštěcí skripty pipeline
src/                # Python balíček (generátory, buildery, extraktory, ML moduly)
templates/          # Šablony OpenFOAM případů
```

### Hlavní moduly (`src/`)

| Modul | Funkce |
|---|---|
| `config.py` | Načítání YAML konfigurací do typed dataclasses |
| `generate_sampling_table.py` | Generování CSV tabulky kombinací profilů a podmínek |
| `sampling.py` | Čtení CSV tabulky do typovaných `BuildCaseRow` záznamů |
| `case_builder.py` | Stavba adresářů případů ze šablony |
| `blockmesh_generator.py` | Procedurální generátor C-grid `blockMeshDict` pro NACA profily |
| `file_editors.py` | Pomocné editory OpenFOAM souborů (`0/U`, `forceCoeffs`) |
| `case_runner.py` | Spouštění OpenFOAM příkazů, logování, čištění |
| `flow_extractor.py` | VTK export, interpolace na pravidelnou mřížku, uložení `.npz` |
| `ml_dataset.py` | PyTorch dataset pro načítání `.npz` souborů |
| `ml_models.py` | Architektury modelů (`SimpleUNet`, fyzikálně informovaná CNN) |
| `ml_training.py` | Trénovací smyčka, masked MSE ztráta |
| `ml_validation.py` | Vykreslování trénovacích křivek |

---

## Konfigurace

Všechna nastavení jsou řízena YAML soubory v `configs/`:

- `paths.yaml` – absolutní cesty k adresářům projektu a výstupům
- `dataset_config.yaml` – parametry vzorkování (profily, AoA, rychlosti) a parametry sítě `blockMesh`
- `solver_config.yaml` – OpenFOAM příkazy, paralelizace, počet workerů
- `ml_models_config.yaml` – volba modelu, hyperparametry, výstupní soubory

---

## Technologie

- **CFD:** OpenFOAM (`simpleFoam`, `blockMesh`, `foamToVTK`)
- **Python:** numpy, scipy, vtk, PyTorch, PyYAML
- **ML:** U-Net surrogate model s volitelnou RANS fyzikální ztrátou


## TO DO
- automatická validace CFD výsledných polí
- automatické vyloučení podezřelých případů
- rozšíření základního datasetu (po přidání automatické validace)
- kalibrace ML modelů na 100 pct fyzikálně čistých datech