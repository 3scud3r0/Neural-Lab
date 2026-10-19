# Neural-Lab · Ouroboros Engine 3.0

Laboratório executável de descoberta de neurônios, programação por bytecode, comunicação discreta, cooperação e morfogênese neural em PyTorch.

O projeto completo está em [ouroboros-engine](ouroboros-engine/README.md), incluindo fontes upstream de NCPS, NEAT-Python e PettingZoo, licenças, checkpoints treinados e relatórios de validação.

```bash
cd ouroboros-engine
bash scripts/install_v2.sh
source .venv/bin/activate
ouroboros dashboard --output validation-v2 --port 8080

# Fábrica de equações neurais, organizada em 16 módulos principais:
ouroboros-research run --output outputs-v3/run-01 --candidates 2048 --finalists 12 --seeds 5
```

[Implementação em 16 módulos](ouroboros-engine/docs/reproduce_v3.md) · [Resultados v3](ouroboros-engine/validation-v3/report.json) · [Somente código em TXT](ouroboros-engine/CODIGO_SOMENTE.txt) · [Resultados dos testes](ouroboros-engine/RESULTADOS_TESTES.txt)
