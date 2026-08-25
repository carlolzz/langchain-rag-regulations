
# Corpus - Sources and Licensing
The source PDFs are located in `data/raw/`, which is **gitignored** and not distributed with this repository. This file records what the corpus is and where each document came from, so that the data can be reconstructed without downloading the documents themselves.

**Retrieved** - 08-2026
**Academic Year** - a.a 2026/2027

## Documents
Six *regolamenti didattici* (degree-programme academic regulations) from Politecnico di Milano, School of Industrial and Information Engineering-

| File (`data/raw/`) | Programme | Url
| --- | --- | --- |
| `regolamento_didattico_automation_control_and_engineering.pdf` | Automation and Control Engineering | 
| `regolamento_didattico_computer_science_and_engineering.pdf` | Computer Science and Engineering | 
| `regolamento_didattico_high_performance_computing_engineering.pdf` | High Performance Computing Engineering | 
| `regolamento_didattico_mathematical_engineering.pdf` | Mathematical Engineering | 
| `regolamento_didattico_mechanical_engineering.pdf` | Mechanical Engineering |
| `regolamento_didattico_telecommunication_engineering.pdf` | Telecommunication Engineering |

## Licensing

*Art.5 of L. 633/1941* states that the law does not apply to *i testi degli atti ufficiali dello stato e delle Amministrazioni pubbliche*. Politecnico di Milano is a public university and a *regolamento didattico* is adopted by rectoral decree, so these texts fall outside copyright protection.
The documents are published openly on polimi.it and are not reproduced here, neither in the repository nor in any distributed artefact.

## Reproducing the corpus

Download the six PDFs from the URLs above into `data/raw/`, keeping the filenames exactly as listed, then run:

```bash
uv run python -m src.ingest
```