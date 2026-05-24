# Project Plan (Current Branch)

## Cíl

Stabilní a reprodukovatelný řetězec:
- blockMesh CFD dataset generation,
- NPZ export,
- ML trénink s konfigurovatelnými parametry.

## Status

### ✅ Hotovo

- přechod na blockMesh-only workflow
- odstranění hlavních hardcoded parametrů do YAML configů
- refactor klíčových modulů (`case_builder`, `case_runner`, `flow_extractor`, `sampling`)
- config-driven ML trénink

### 🔄 Průběžně

- ladění kvality mřížky a doménové délky wake
- kontrola numerické stability v celé sadě case

## Další kroky

1. Automatická validace config souborů (schema + smysluplné rozsahy)
2. Smoke test pipeline (build -> run -> extract) nad malým subsetem
3. Přidat report metrik po běhu (konvergence, fail-rate, runtime distribuce)
4. Rozšířit QA pro NPZ dataset (NaN/Inf, fyzikální limity)
5. Iterace ML experimentů s verzováním konfigurací

## Rizika

- nekonzistentní config mezi prostředími (Windows/OpenFOAM paths)
- změny v mesh parametrech mohou zhoršit stabilitu solveru
- vysoké nároky na výpočetní čas při větším sweepu

## Doporučení

- držet všechny experimentální změny pouze přes YAML configy,
- vždy logovat config snapshot spolu s výsledky běhu/tréninku,
- zavést minimální regression test na 1–3 reprezentativních case.