# Execução validada da v3

- 16 módulos principais, com testes separados.
- 85 testes automatizados passaram, sem falhas ou testes pulados.
- 2.048 equações treinadas na triagem; 12 finalistas receberam mais treino.
- 1.941 assinaturas distintas nas sondagens numéricas de comportamento.
- Cinco sementes novas por modelo, seis modelos: 30 treinamentos independentes.
- Quatro tarefas sintéticas; teste e transferência registrados por tarefa.
- Seis arquiteturas heterogêneas avaliadas em validação na integração.
- Crescimento/poda 6 → 9 → 7, preservando valores e momentos Adam compatíveis.
- Política individual, memória, herança, destilação, construção, ferramentas e
  agenda de pesquisa foram exercitadas; os episódios curtos não provam competência.

| Modelo | MSE médio final — menor é melhor |
|---|---:|
| Candidato descoberto | 0,159060 |
| Ablação `tanh(x)` | 0,163315 |
| MLP | 0,240451 |
| RNN | 0,154328 |
| GRU | 0,160234 |
| LSTM | 0,167143 |

**Promoção rejeitada:** não houve melhoria suficientemente consistente sobre
todos os controles. O candidato permanece como resultado experimental.

A execução inicial sem cache levou 73,275 segundos na CPU disponível. A verificação
posterior recuperou a triagem do SQLite e repetiu a comparação independente e as
integrações em 4,705 segundos. Esses tempos estão em `execution_history` no relatório;
o campo `seconds` é o tempo da última execução com cache.

## Arquivos

- `report.json`: configurações, métricas, comparações, intervenções e limitações.
- `config.json`: configuração da busca entregue.
- `experiments.sqlite`: 2.048 equações, 2.048 avaliações de triagem e 12 avaliações
  com treinamento adicional. Colunas da tabela `observations`: `id`, `stage`, `payload`.
- `winner-equation.json`: árvore da equação escolhida na validação.
- `winner.pt`: modelo e Adam do candidato treinado, carregável com `weights_only=True`.
- `inventory.json`: hashes das 16 fontes e verificação do checkpoint.
- `code-export.json`: inventário do TXT contendo somente código.
- `archive-check.json`: verificações do pacote extraído.

Os pesos de todos os candidatos intermediários ficam somente na execução local
em `outputs-v3/release/search/weights/`. Para refazer os experimentos a partir da
distribuição, use um diretório novo e siga `docs/reproduce_v3.md`.

O resultado detalhado da suíte está em `../RESULTADOS_TESTES.txt`, separado das
fontes de `../CODIGO_SOMENTE.txt`. As licenças upstream estão em `../third_party/`.
