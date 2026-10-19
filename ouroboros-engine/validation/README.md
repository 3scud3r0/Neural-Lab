# Evidências de execução

Os arquivos desta pasta foram produzidos durante a validação em CPU:

- `tests.txt`: saída da suíte de 13 testes; todos passaram.
- `programmer.pt` e `programmer.json`: bootstrap de 150 épocas, com seis tarefas corretas em novos exemplos numéricos das mesmas tarefas.
- `demo/`: execução de 300 passos, 24 agentes sociais, um agente programador e 12 tentativas de programação corretas. Inclui métricas, programas gerados e snapshot dos pesos.
- `dashboard-desktop.png` e `dashboard-mobile.png`: capturas do painel em Chromium, com métricas dessa execução.
- `report.json`: resumo da validação. CUDA não foi testada porque o ambiente não oferece GPU.

Para explorar esses dados sem treinar novamente, execute na raiz do projeto:

```bash
python scripts/dashboard.py --output validation/demo
```

Comandos utilizados para validar a implementação (escolha uma pasta de saída nova ao repetir):

```bash
python -m unittest discover -s tests -v
ouroboros-bootstrap --output validation/programmer.pt
ouroboros-run --steps 300 --device cpu --checkpoint validation/programmer.pt --output outputs/final-run
ouroboros-dashboard --output outputs/final-run --port 8081
```

O painel também foi verificado automaticamente em Chromium, em dimensões desktop e mobile: nenhum erro JavaScript, status de execução concluída, 32 barras de ativação e ausência de transbordamento horizontal em mobile. Resultados de correção não comprovam novidade algorítmica ou capacidade de programação geral.
