# Ouroboros Engine

Laboratório experimental de agentes adaptativos em **PyTorch**, baseado na arquitetura da conversa fornecida. Inclui ecossistema 2D, recorrência temporal, plasticidade local, memória rápida inspirada em TTT, homeostase numérica, comunicação e uma programadora de expressões aritméticas com feedback de execução.

**É um protótipo funcional de pesquisa.** Não é uma inteligência geral, não demonstra consciência, não inventa linguagens e não promete superar Rust ou acelerar ray tracing. A implementação e suas medições permitem investigar hipóteses concretas antes de ampliar o projeto.

## Começar no Linux

Requisitos: Python 3.10 ou superior, `venv`, acesso aos índices oficiais de pacotes. Testado com Python 3.12.14, PyTorch 2.6.0+cpu e PyYAML 6.0.3. CUDA é opcional. O executor utiliza `resource` do Linux; Windows nativo não é suportado. Use WSL2 no Windows.

Na pasta `ouroboros-engine`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
```

O índice de CPU evita baixar bibliotecas CUDA desnecessárias. Para GPU, instale primeiro a distribuição PyTorch compatível com seu driver, seguindo o seletor oficial em <https://pytorch.org/get-started/locally/>, e depois execute `pip install -e .`. Este projeto não contém extensões CUDA próprias. `--device auto` usa CUDA quando disponível; `--device cuda` falha explicitamente se não houver GPU utilizável.

### Treinar e executar

```bash
# Ensina seis tarefas ao seletor de expressões; grava pesos e relatório.
python scripts/bootstrap_programmer.py

# Executa 300 passos. --pace torna a observação ao vivo mais confortável.
python scripts/run_ecosystem.py --steps 300 --checkpoint outputs/programmer.pt --output outputs/demo --pace 0.05

# Em outro terminal, com o mesmo ambiente virtual ativado:
python scripts/dashboard.py --output outputs/demo --port 8080
```

O painel atende na porta 8080, na interface local por padrão. Em uma máquina própria, abra essa porta no navegador; em um serviço de nuvem, utilize o mecanismo de acesso a portas da plataforma, quando disponível. O painel também consegue apresentar uma execução já concluída. Não exige Streamlit, CDN, APIs pagas ou chaves de acesso.

Para repetir a simulação, escolha uma nova pasta, por exemplo `--output outputs/demo-2`, e aponte o painel para ela. A simulação **recusa diretórios não vazios**, para preservar resultados anteriores. Sem `--checkpoint`, a programadora começa com pesos aleatórios e recebe reforço durante a execução. Com bootstrap, começa já treinada nas seis tarefas.

Os mesmos comandos estão disponíveis como `ouroboros-bootstrap`, `ouroboros-run` e `ouroboros-dashboard` após a instalação editável. Execute os comandos de exemplo na raiz desta pasta. Em tarefas na nuvem, use o checkout existente: não é necessário criar um Git worktree.

## O que está implementado

| Componente | Funcionamento real |
|---|---|
| `primitives/liquid_cell.py` | Relaxação exponencial em tempo contínuo com constante de tempo aprendida; inspirada em células líquidas, não implementação canônica de CfC. |
| `primitives/plastic_synapse.py` | Pesos treináveis mais traço Hebbiano individual, com decaimento, modulação e limites. |
| `primitives/spiking_gate.py` | Gate de limiar com gradiente substituto; fração de disparos medida. |
| `primitives/ttt_layer.py` | Atualização analítica por SGD de preditor linear rápido, separado por organismo. |
| `physiology/` | Energia, custos de movimento/sinal, estresse, entropia de ativações e modulação da plasticidade. |
| `core/dynamic_brain.py` | Integração recorrente das primitivas, previsão sensorial, ação e emissão de sinais. |
| `core/meta_plasticity.py` | Operações PyTorch em lote, compatíveis com CPU/CUDA; sem kernels personalizados. |
| `agents/social_agent.py` | Organismos sociais vetorizados: pesos lentos compartilhados e memórias rápidas individuais. |
| `agents/programmer_agent.py` | Organismo individual com decoder de ASTs aritméticas, seleção estocástica e REINFORCE. |
| `environment/` | Recursos limitados, competição, reposição de estoque, atrito, obstáculos móveis e sinais atenuados por distância. |
| `sandbox/` | Interpretação de DSL restrita em subprocesso, limites de CPU/RAM/tempo e retorno de correção. |
| `scripts/` | Bootstrap supervisionado, simulação com aprendizado online e painel HTTP de métricas reais. |

O loop do ecossistema é sequencial e determinístico na CPU com a mesma semente; o painel pode rodar em um processo independente e consultar snapshots atômicos. CUDA pode introduzir diferenças numéricas. A interface usa HTTP com atualização periódica, não WebSockets.

### Aprendizado

1. Os agentes observam sua posição, direção/distância ao recurso mais próximo, energia, estresse e mensagens recebidas.
2. O cérebro produz previsão da próxima observação, ações estocásticas e sinais contínuos.
3. O mundo recompensa colheita e aproximação dos recursos; movimento e colisões têm custo.
4. Adam otimiza previsão, política motora com REINFORCE, política de comunicação com recompensa coletiva imediata e regularização metabólica. Não há garantia de que essa recompensa ensine comunicação útil.
5. A plasticidade Hebbiana e a memória rápida TTT atualizam estados locais durante cada passo.
6. Quando a energia acaba, o organismo é contado como perdido e reposto, com memória rápida limpa. Os pesos lentos compartilhados permanecem.

As perdas estão explícitas em `ouroboros/simulation.py`. O módulo `free_energy.py` calcula erro preditivo com regularização; **não implementa uma ELBO nem comprova o Princípio da Energia Livre**. A entropia mostrada é das ativações, não uma medida validada de entropia sináptica. Energia/estresse são variáveis numéricas, sem alegação de sentimentos.

### Escopo da programadora

O bootstrap treina um classificador de **11 expressões predefinidas** para seis tarefas identificadas por vetores de contexto. O estado do cérebro modula a entrada desse decoder; o feedback altera a política e a homeostase. Não há modelo de linguagem pré-treinado, geração livre de tokens, compreensão de instruções em linguagem natural nem autoalteração do código do projeto.

O dataset em `data/bootstrap_code/tasks.json` separa casos de treino e de validação numérica. A avaliação verifica novos números nas mesmas tarefas conhecidas: **não testa generalização para algoritmos inéditos**. A execução online usa os casos de validação para recompensa, portanto esses casos deixam de ser um conjunto independente de avaliação após o aprendizado online.

Cada tentativa gera um pequeno arquivo `solve(x, y)` e um JSON com expressão, recompensa, correção, tempo e memória. A correção tem prioridade: uma saída vazia, erro ou função constante incorreta não ganha recompensa por ser rápida. Ainda não há objetivo de otimização de tempo de execução nem comparação de desempenho entre algoritmos.

### Limites do executor

A DSL aceita somente números limitados, `x`, `y`, `+`, `-`, `*`, `/`, sinais unários e chamadas `abs`, `min`, `max` com aridade fixa. Não utiliza `eval` ou `exec`. Imports, atributos, arquivos, rede, loops, comprehensions e potência são rejeitados. Há limites de tamanho da AST, profundidade e quantidade de casos.

O subprocesso recebe um ambiente mínimo sem credenciais herdadas e limites Linux de memória, CPU e arquivos. O tempo total é limitado pelo processo pai. `elapsed_ms` mede interpretação; `wall_ms` inclui inicialização do processo; `memory_kb` é o pico RSS pós-exec do worker, medido por `VmHWM` em `/proc/self/status` (e não pelo pico eventualmente herdado do processo pai). Esses valores não são benchmarks de algoritmos.

**Isto não é um sandbox para Python arbitrário, C ou assembly.** Não use o executor como serviço público para código livre. Para ampliar esse escopo, será preciso isolamento de SO apropriado, como contêineres com seccomp, filesystem e rede restritos. Os exemplos C/ASM são apenas referências e não são executados pelo sistema.

## Resultados gravados

```text
outputs/
  programmer.pt             # Pesos do bootstrap
  programmer.json           # Avaliação das seis tarefas
  demo/
    config.json             # Configuração efetiva e versões
    metrics.jsonl           # Um registro por passo
    latest.json             # Snapshot atômico para o painel
    summary.json            # Métricas finais
    ecosystem.pt            # Pesos/estados finais para análise
    programs/
      attempt_0001.py        # Função gerada a partir da DSL
      attempt_0001.json      # Resultado da tentativa
```

`ecosystem.pt` é um snapshot de análise, **não um checkpoint de retomada exata**: não inclui todos os estados do mundo, RNGs e otimizadores. A retomada de simulação não está implementada. Carregue checkpoints locais confiáveis com `weights_only=True`, como fazem os scripts.

O ZIP inclui `validation/` com resultados de uma execução real em CPU, relatório de testes, bootstrap treinado e amostras de programas. A pasta `outputs/` não é necessária para começar. Os pesos de bootstrap entregues podem ser utilizados com `--checkpoint validation/programmer.pt`.

Para visualizar a execução entregue, use `python scripts/dashboard.py --output validation/demo`. Para criar novamente o ZIP com os fontes e as evidências, use `python scripts/package_project.py`.

## Configuração e extensão

- `configs/ecosystem_config.yaml`: população, dispositivo, passos, geometria, recursos, metabolismo e frequência da programadora.
- `configs/neuron_primitives.yaml`: constantes de tempo, limiar de disparo, taxas e decaimento das memórias rápidas.
- `configs/programmer_bootstrap.yaml`: semente, épocas, taxa e dataset de bootstrap. O contexto de seis dimensões é fixo neste protótipo; `syntax_penalty` documenta a penalidade padrão de falha do executor.

Para mudar a gramática, atualize `EXPRESSIONS`, os exemplos e os testes e treine um novo checkpoint. Para avaliar emergência ou desempenho, defina baselines, tarefas não vistas, métricas independentes e várias sementes. A topologia atual é fixa: as conexões mudam de peso, mas a rede não cria/remove neurônios. O gate esparso continua usando operações densas e não garante economia de GPU ou 90% de esparsidade.

## Testes

```bash
python -m unittest discover -s tests -v
```

A suíte verifica gradientes e dinâmica temporal, memórias individuais, comunicação por distância, conservação de recursos, obstáculos, rejeição de construções proibidas, timeout, feedback, repetibilidade e artefatos do ciclo integrado, além dos endpoints HTTP do painel. A validação CUDA depende de hardware disponível e não foi realizada no ambiente de entrega.

## Exportação de código-fonte

`CODIGO_COMPLETO.txt` contém as fontes, configurações e documentação reunidas num único arquivo UTF-8. Gere novamente com `python scripts/export_source_txt.py`. Os pesos PyTorch e as capturas PNG são binários separados incluídos na distribuição ZIP e nos artefatos de validação.
