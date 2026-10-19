# Ouroboros 3: laboratório em 16 arquivos principais

Esta versão implementa uma infraestrutura de pesquisa pequena e executável para
gerar, treinar, comparar e rejeitar ou promover equações neurais. Os 150 caminhos
do planejamento anterior foram agrupados por responsabilidade nos 16 módulos
abaixo. O agrupamento não transforma o roteiro inteiro em um sistema de inteligência
geral; cada capacidade tem um alcance explícito.

## Instalação e execução

Linux/WSL2, Python 3.11 ou 3.12. Não há dependências adicionais às da v2.

```bash
cd ouroboros-engine
bash scripts/install_v2.sh
source .venv/bin/activate
ouroboros-research --help

# Executar milhares de candidatos com orçamento explícito:
ouroboros-research run --output outputs-v3/reproducao \
  --candidates 2048 --screening-steps 3 --promotion-steps 24 \
  --finalists 12 --seeds 5 --seed 42 --max-seconds 1800

# Executar todos os testes, incluindo v1 e v2:
ouroboros-research test --output RESULTADOS_TESTES.txt \
  --report validation-v3/report.json

# Exportar somente fontes; resultados e documentação ficam fora:
ouroboros-research export --output CODIGO_SOMENTE.txt

# Painel de leitura dos resultados:
ouroboros-research dashboard --output validation-v3 --port 8083
```

Também é possível usar `python -m ouroboros.v3.application` no lugar de
`ouroboros-research`. A execução da v3 foi validada em CPU; CUDA não faz parte
deste fluxo. A instalação original precisa de rede para as dependências.

## Ordem dos arquivos

Todos estão em `ouroboros/v3/`. Existem exatamente 16 arquivos Python nesse
diretório. Os testes são separados, em `tests/test_v3.py`.

| Ordem | Arquivo | Responsabilidade implementada |
|---|---|---|
| 1 | `__init__.py` | Contratos, identidade de candidatos e configuração validada. |
| 2 | `experiments.py` | SQLite, artefatos atômicos, orçamento, RNG e checkpoint de modelo/Adam. |
| 3 | `neuron_dsl.py` | Árvores de equações tipadas implicitamente por forma, validação e execução diferenciável sem Python arbitrário. |
| 4 | `neurons.py` | Oito motivos iniciais; estados recorrentes, memória rápida, adaptação, fase e opção de spikes com gradiente substituto. |
| 5 | `plasticity.py` | Hebb, STDP, elegibilidade e modulação aprendida, com limites de atualização. |
| 6 | `benchmarks.py` | Atraso, integração, troca de memória e filtragem de ruído; partições e quatro controles. |
| 7 | `search.py` | Mutação/recombinação de equações, seleção, treinamento adicional e arquivo diverso. |
| 8 | `evaluation.py` | Repetições independentes, ablação, transferência e decisão de promoção. |
| 9 | `architecture.py` | Regiões heterogêneas, busca conjunta aleatória de topologia/equações e crescimento/poda com migração de estados. |
| 10 | `memory.py` | Memória episódica limitada, recuperação por similaridade, replay e penalidade de consolidação. |
| 11 | `world.py` | Mundo parcialmente observado de coleta, fabricação de ferramenta e construção; API ParallelEnv. |
| 12 | `society.py` | Políticas e memórias individuais, REINFORCE, herança, destilação e integração com o jogo de comunicação livre da v2. |
| 13 | `programming.py` | Contratos de funções, dependências, compilação restrita, diagnóstico por traces, síntese e composição de ferramentas da VM existente. |
| 14 | `selfmod.py` | Propostas de equações por hash, contrato externo, avaliação, histórico e rollback. |
| 15 | `research.py` | Agenda UCB para hipóteses declaradas, orçamento e evidências. |
| 16 | `application.py` | CLI, integração, relatórios, painel de leitura, exportação de fontes e execução dos testes. |

## O que a busca mede

Uma equação recebe preativação (`x`), estado anterior (`h`), memória rápida (`m`),
adaptação (`a`) e fase (`p`). Os operadores incluem mistura com porta, seno,
tangente hiperbólica, sigmoid, softsign e aritmética limitada. Árvores têm limites
de profundidade, tamanho e constantes. Não podem importar módulos, executar
comandos, alterar o avaliador ou criar loops na DSL.

O envelope do neurônio mantém atualizações fixas para memória e adaptação;
a busca desta entrega evolui a equação de dinâmica, não descobre automaticamente
todas as regras de aprendizagem. As regras de plasticidade são componentes
funcionais separados, com testes de comportamento e gradientes.

Cada candidato recebe três atualizações Adam e é avaliado em validação nas quatro
tarefas. Os 12 melhores recebem mais 24 atualizações, preservando o otimizador.
A seleção usa exclusivamente validação. Só depois o vencedor é reconstruído e
treinado do zero com cinco novas sementes, junto dos controles. Cada repetição
recebe 27 atualizações. Quantidades de parâmetros e tempos reais são registrados;
isso não equivale a igualdade de FLOPs, custo total de busca ou ajuste ótimo de
hiperparâmetros para cada modelo.

Os controles são MLP, RNN, GRU, LSTM e uma ablação com equação `tanh(x)` no mesmo
envelope do candidato. Todos têm largura 12. O relatório inclui tarefa por tarefa,
erro médio quadrático, transferência para sequências mais longas e perturbações.
As quatro tarefas são sintéticas; não representam toda capacidade cognitiva.

As 2.048 equações são distintas por estrutura serializada. Expressões diferentes
podem implementar a mesma função. O relatório também conta assinaturas numéricas
em 17 estados de sondagem; esse descritor aproxima diversidade comportamental,
sem provar equivalência ou diferença funcional em todos os estados.

O critério de promoção exige margem de melhoria sobre todos os controles,
considerando diferenças pareadas e uma margem de incerteza de três erros padrão.
É um filtro operacional conservador, não um teste estatístico exato que certifique
uma descoberta com cinco amostras. A seleção de milhares de candidatos reforça a
necessidade de avaliações externas adicionais antes de qualquer alegação ampla.

## Evidências e retomada

`validation-v3/report.json` contém a execução medida; `experiments.sqlite` contém
as observações de todos os candidatos e suas equações. `winner.pt` preserva o
checkpoint escolhido na validação. A reprodução exata de um resultado numérico
depende também das versões, CPU e bibliotecas registradas.

Para retomar uma busca local, repita o comando com o mesmo diretório e configuração.
Os candidatos concluídos são recuperados do SQLite. Um candidato interrompido
durante o treinamento é reiniciado; a busca não promete retomada no meio de uma
instrução. Os checkpoints locais de triagem ficam em `outputs-v3/`, fora da entrega;
o arquivo entregue preserva o vencedor e as evidências, não todos esses pesos.

O checkpoint genérico de modelo/otimizador tem teste de continuação exata em CPU.
A migração estrutural suporta Adam de um grupo e cria uma fronteira de gradiente:
preserva valores compatíveis e momentos, mas descarta o grafo antigo.

## Alcance das integrações

O mundo é uma grade pequena, com ações discretas e regras explícitas. Há recursos
consumíveis, ferramenta que aumenta coleta, construção, custo energético e sinais
privados. Os testes oficiais PettingZoo verificam o contrato da API. Os poucos
episódios de integração verificam o caminho de aprendizagem; não demonstram uma
sociedade competente, linguagem espontânea composicional ou divisão de trabalho.

O programador reaproveita o decoder treinado, o compilador, a busca genética, arrays,
loops e chamadas da v2. O planejamento de módulos valida dependências e ciclos;
não oferece Python irrestrito, um novo modelo aberto de código nem um compilador
de aplicações arbitrárias. A autoalteração abrange equações e ferramentas da VM,
com avaliador externo; não reescreve livremente o host.

O pesquisador escolhe entre hipóteses declaradas por UCB. A memória e a consolidação
estão disponíveis como componentes; esta entrega não demonstra aprendizagem
contínua de longo prazo em todos os agentes. Os motivos chamados dendrítico e
oscilatório são simplificações computacionais, não simulações biológicas completas.

## Entregáveis de texto

`CODIGO_SOMENTE.txt` concatena fontes próprias, testes, scripts, exemplos e fontes
upstream. As linhas `# SOURCE:` são comentários que identificam os arquivos.
Não contém relatórios, logs ou documentação externa às próprias fontes. É uma
exportação textual de fontes de linguagens diferentes, não um executável compilado.
As licenças originais continuam nos arquivos de `third_party/` e nos avisos da
distribuição; o TXT isolado não substitui o pacote licenciado.

`RESULTADOS_TESTES.txt` contém comando, código de saída, resultados individuais da
suíte e o relatório de pesquisa separado do código. `CODIGO_COMPLETO.txt` permanece
como artefato histórico da v2 e inclui documentação; use o novo TXT para esta entrega.
