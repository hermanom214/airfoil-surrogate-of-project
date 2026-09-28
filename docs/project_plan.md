# Project Status and Plan

## Hotovo

- procedurální blockMesh-only C-grid pipeline;
- config-driven sampling, mesh build a OpenFOAM runner;
- per-case divergence guard, logy, VTK conversion, archivace a cleanup;
- extrakce polí na pravidelnou mřížku a dataset QA;
- spatial modely `simple_unet`, `rans_pinn` a scalar `clcd_mlp`;
- jednotný fixed-test experiment protocol;
- holdout nebo K-fold/NACA-group CV;
- randomized/manual coarse hyperparameter search;
- training-only normalizace bez leakage z validation/test dat;
- oddělené checkpointy a metadata pro každý model;
- evaluace fixed test setu pro spatial i Cl/Cd větev;
- CPU/CUDA device selection a izolovaný smoke-test output.

## Aktuální omezení

- absolutní Windows/OpenFOAM cesty v `paths.yaml` musí odpovídat lokálnímu checkoutu;
- build skript vždy regeneruje celou sampling table a smaže předchozí case adresáře;
- build a CFD runner nemají malý subset/smoke režim;
- extraction grid a počet extraction workers jsou hardcoded ve skriptu;
- case archivace nepřepisuje existující `data/flow_fields/<case_id>`;
- `run_status.csv` leží v simulačním kořeni, ze kterého se úspěšné cases průběžně odstraňují;
- názvy evaluation adresářů používají suffix `_validation`, i když nové checkpointy vyhodnocují fixed test set;
- některé starší pomocné moduly a komentáře zůstávají kvůli kompatibilitě.

## Doporučené další kroky

1. Přidat validační preflight pro cesty, YAML hodnoty, šablonu a dostupnost OpenFOAM/CUDA.
2. Přidat bezpečný `--case-id`, `--limit` nebo manifest subset režim pro build/run/extract.
3. Přesunout extraction grid, workers a divergence thresholds do konfigurace.
4. Přidat explicitní politiku `--overwrite/--resume` pro case archiv a NPZ.
5. Vytvořit jeden end-to-end smoke test od build přes CFD až po evaluaci.
6. Ukládat snapshot všech použitých konfigurací ke každému experimentu.
7. Sjednotit názvy test/evaluation výstupů a zachovat čtení legacy cest.
8. Dále kalibrovat mesh robustnost a physics-loss váhu na reprezentativních cases.
9. Rozpracovat PINN model

## Provozní pravidla

- Před buildem zazálohovat nebo přesunout ručně cenné cases ze spravovaných `blockmesh_cases` kořenů.
- Před opakovaným CFD během rozhodnout, zda zachovat, přesunout nebo odstranit existující cílový case archiv.
- Po CFD zkontrolovat `run_status.csv`, logy a inspection výsledky před tréninkem.
- Neměnit fixed test split uprostřed série porovnávaných experimentů.
- Pro rychlou kontrolu ML použít `--smoke-test`; výsledky nepovažovat za produkční benchmark.
