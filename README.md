# Briscola RL Agent

Progetto per il corso di Reinforcement Learning 2025/26 (UniPD): agente RL che
gioca a Briscola contro lo `ScriptedAIAgent` fornito.

Il codice vive nel package `brimarl_masked/` (il nome è vincolato: tutti gli
import interni usano `brimarl_masked.*`). Il README originale del progetto,
con la descrizione dei moduli, è in [`brimarl_masked/README.md`](brimarl_masked/README.md).

## Setup

```bash
conda create -n RL python=3.12 -y
conda activate RL
pip install tensorflow numpy matplotlib tqdm
```

## Utilizzo

Eseguire sempre dalla **radice del repository** (la cartella che contiene
`brimarl_masked/`), in forma modulo:

```bash
# valutazione
python -m brimarl_masked.scritps.evaluate

# training
python -m brimarl_masked.main.train_best_response
```

In alternativa, per lanciare un file direttamente: `PYTHONPATH=. python brimarl_masked/main/train_best_response.py`.

Nota: la cartella `scritps` si chiama davvero così (typo del codice originale,
usato dagli import).
