
# Corpus - Sources and Licensing

The source PDFs are located in `data/raw/`, which is **gitignored** and not distributed with this repository. This file records what the corpus is and where each document came from, so that the data can be reconstructed without downloading the documents themselves.

**Retrieved** - 09-2026

## Documents

Six *regolamenti edilizi comunali* from 6 different Italian cities.

| File (`data/raw/`) | Programme | Url
| --- | --- | --- |
| `milano_regolamento_edilizio.pdf` | Regolamento edilizio comunale di Milano | [Shortened Link](https://www.comune.milano.it/amministrazione/regolamenti-comunali/e-f/edilizio) |
| `genova_regolamento_edilizio.pdf` | Regolamento edilizio comunale di Genova | [Shortened Link](https://www2.comune.genova.it/content/regolamento-edilizio-comunale) |
| `firenze_regolamento_edilizio.pdf` | Regolamento edilizio comunale di Firenze | [Shortened Link](https://ediliziaurbanistica.comune.fi.it/edilizia/atti_normativa/regolamento_edilizio.html) |
| `padova_regolamento_edilizio.pdf` | Regolamento edilizio comunale di Padova | [Shortened Link](https://www.comune.padova.it/amministrazione/documenti-e-dati/atto-normativo/regolamento-edilizio) |
| `torino_regolamento_edilizio.pdf` | Regolamento edilizio comunale di Torino | [Shortened Link](https://www.comune.torino.it/amministrazione/documenti-dati/documenti/n-381-regolamento-edilizio) |
| `verona_regolamento_edilizio.pdf` | Regolamento edilizio comunale di Verona | [Shortened Link](https://www.comune.verona.it/Novita/Notizie/Regolamento-Edilizio) |

## Licensing

*Art.5 of L. 633/1941* states that the law does not apply to "*i testi degli atti ufficiali dello stato e delle Amministrazioni pubbliche*". The documents are published openly on the listed websites, and are not reproduced here, neither in the repository nor in any distributed artefact.

## Reproducing the corpus

Download the six PDFs from the URLs above into `data/raw/`, keeping the filenames exactly as listed, then run:

```bash
uv run python -m src.ingest
```