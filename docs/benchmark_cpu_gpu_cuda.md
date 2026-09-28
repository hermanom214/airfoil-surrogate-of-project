# Historický CPU/GPU benchmark

Naměřený čas tří epoch U-Netu:

| Device | 3 epochy | Přibližně 1 epocha |
|---|---:|---:|
| CPU | 54 min 15 s (3255 s) | 18 min 5 s |
| GTX 1070 Ti | 5 min 12 s (312 s) | 1 min 44 s |

Naměřený speedup byl `10.43×`. V tomto běhu tedy načítání dat z HDD nebylo dominantním bottleneckem.

## Jak benchmark zopakovat

Aktuální trénovací skript dovoluje shodný počet epoch i explicitní device:

```powershell
python -m scripts.train_ml_model --model simple_unet --device cpu --epochs 3 --single-run --search-strategy none
python -m scripts.train_ml_model --model simple_unet --device cuda --epochs 3 --single-run --search-strategy none
```

Oba běhy musí používat stejný dataset, fixed split, hyperparametry a softwarové prostředí. Produkční checkpoint má pro oba příkazy stejnou cestu, proto je nutné první sadu artefaktů před druhým během uchovat mimo modelový adresář. Časy výše jsou historické; nejsou garantovaným výkonem aktuálního kódu ani jiného hardwaru.
