# Ouroboros Engine 2.0

Laboratório executável de **programação composicional, comunicação discreta, cooperação multiagente e morfogênese neural**. A versão 2 substitui o classificador de 11 expressões por geração autorregressiva de bytecode, adiciona síntese genética por exemplos, redes com dimensões modificáveis e integra código real de três projetos do GitHub.

O ZIP inclui fontes, bibliotecas upstream e suas licenças, modelos treinados, programas gerados, relatórios, testes e um TXT consolidado. A execução padrão usa CPU; os modelos PyTorch também aceitam CUDA quando disponível. Não utiliza APIs de LLM ou chaves pagas.

## Começar

Use **Linux ou WSL2, Python 3.11/3.12**. Na pasta `ouroboros-engine` extraída do ZIP:

```bash
bash scripts/install_v2.sh
source .venv/bin/activate
ouroboros doctor

# Abrir o painel com os resultados e pesos já entregues:
ouroboros dashboard --output validation-v2 --port 8080
```

O servidor atende a interface local na porta 8080. Em uma máquina própria, abra essa porta no navegador; em serviços de nuvem, utilize o mecanismo de acesso a portas fornecido pela plataforma. O painel permite executar assembly da VM ou a DSL imperativa, inspecionar programas gerados e comparar os experimentos. Não depende de CDN, WebSockets externos ou serviços pagos.

A instalação usa os fontes incluídos em `third_party/`. Internet é necessária para PyTorch, NumPy e as demais dependências de distribuição. Os binários desses pacotes não foram embutidos no ZIP. O projeto é instalado em modo editável: mantenha a pasta completa, pois datasets, configurações e fontes verificáveis fazem parte do laboratório.

Para CUDA, configure uma distribuição PyTorch 2.6.0 compatível com seu driver pelo seletor oficial em <https://pytorch.org/get-started/locally/>. O instalador também aceita `OUROBOROS_TORCH_INDEX` apontando para o índice oficial correspondente. CUDA não estava disponível na máquina de validação desta entrega.

## O que mudou

| Abismo da versão anterior | Implementação 2.0 |
|---|---|
| Classificador de programas prontos | Decoder GRU emite opcodes e operandos; recebe exemplos numéricos, sem um ID de solução. Suporta comprimentos e fluxos de controle diferentes. |
| Sem composição aberta | Busca genética combina árvores por mutação/cruzamento. Um arquivo JSON pode especificar novas tarefas sem incluir a solução. |
| Sem loops e memória operacional | VM com `INPUT`, `LOAD`, `STORE`, `ALLOC`, `READ`, `WRITE`, aritmética, condições, saltos, `CALL`, `RET` e saída explícita. |
| Comunicação contínua sem tarefa informativa | Jogo referencial com alvo privado, candidatos embaralhados e símbolos discretos; avaliações com silêncio, ruído e mensagens embaralhadas. |
| Topologia fixa | Crescimento e poda alocam Parameters de dimensões diferentes; conexões são removidas/adicionadas, com preservação dos pesos sobreviventes. |
| Mundo sem cooperação necessária | Chaves iniciais irreversíveis e colheita que exige dois papéis segurando o recurso correto, com inércia, atrito, obstáculos, predador e energia. |
| Ferramentas imutáveis | Biblioteca executável de bytecode com revisão, validação, exigência opcional de melhoria e rollback. |
| Peças externas apenas sugeridas | NCPS, NEAT-Python e PettingZoo incluídos por commit, com licenças e manifesto SHA-256, e usados pelos experimentos. |

## Resultados entregues

Estes valores estão em `validation-v2/report.json`, semente **42**, CPU, perfil completo. Os checkpoints podem ser usados sem repetir o treino.

| Avaliação | Resultado registrado |
|---|---|
| Programas neurais | **11/11 famílias**, 112/112 casos finais corretos; incluem fatorial, soma em loop, arrays, contagem condicional e MDC. São algoritmos demonstrados no bootstrap, avaliados em novas entradas. |
| Novos conjuntos de exemplos públicos | **53/55** contextos novos interpretados corretamente nas famílias treinadas; duas falhas ficam registradas. |
| Síntese genética | Composição de `3*x + 2*y - 1` construída sem programa inicial de solução, correta nos 12 testes finais. |
| Nova especificação JSON | Busca sobre `data/v2/tasks/novel_affine.json`, sem fonte de referência, correta em 4/4 casos finais. |
| Revisão executável | Ferramenta de soma passou à revisão 2; a chamada real de `17 + (-9)` produziu `8`. |
| Comunicação livre | 93,6% nos conceitos treinados e **20,5%** nas combinações reservadas: não houve boa generalização composicional. |
| Comunicação com estrutura por atributo | **100%** nos 1.024 testes de combinações reservadas; **24,7%** quando silenciada. A posição dos atributos é um viés explícito; os símbolos são aprendidos. |
| Cooperação física | **97/128 (75,8%)** com canal íntegro; **25/128 (19,5%)** silenciado; **0/128** com um agente sem cooperar. |
| Morfogênese | No experimento estrutural: **12 → 15 → 18 → 15** neurônios, incluindo adição e remoção reais de parâmetros. |
| NEAT upstream | Variação estrutural observada; vencedor com 5 nós e 13 conexões habilitadas. MSE reservado de controle: 0,282. |
| Testes automatizados | **41 testes passaram**, incluindo a API oficial ParallelEnv, limites da VM, isolamento de informação, gradientes e retomada da comunicação. |

A comparação da arena usa os mesmos 128 episódios por intervenção. Os JSONs incluem intervalos de Wilson de 95%. Esses intervalos descrevem a avaliação da política treinada; não substituem replicação do treinamento em múltiplas sementes. O experimento integrado desta entrega levou aproximadamente 30 segundos na CPU disponível, sem incluir instalação e inspeção no navegador.

O protocolo livre e o fatorado permanecem separados para tornar visível a diferença entre código aprendido sem posições semânticas fixas e uma composição auxiliada por estrutura. Os resultados não comprovam consciência, inteligência geral, gramática humana emergente, descoberta científica inédita ou otimizações de ray tracing.

## Executar novamente

```bash
# Pipeline completo; retorna erro se um dos critérios de integração falhar:
ouroboros experiment --output outputs-v2/run-01 --profile full --require-success

# Em outro terminal, para acompanhar:
ouroboros dashboard --output outputs-v2/run-01 --port 8080

# Passagem curta pelo pipeline; pode não haver convergência dos modelos:
ouroboros experiment --output outputs-v2/quick-01 --profile quick

# Todos os testes do projeto:
python -m unittest discover -s tests -v
```

Sempre use uma pasta de saída nova. A CLI recusa pastas não vazias para preservar resultados. Os perfis têm orçamentos finitos. O bootstrap neural encerra cedo quando todas as famílias passam na validação; o teste final é usado depois da seleção. A execução em CPU com a mesma configuração e semente é reproduzível; CUDA pode ter diferenças numéricas.

Cada tarefa na nuvem já possui seu checkout isolado. Use o checkout existente; não é necessário criar um Git worktree.

## Programação por exemplos e execução

```bash
# Uma nova especificação pública, sem fonte da solução:
ouroboros synthesize --examples data/v2/tasks/novel_affine.json \
  --output outputs-v2/my-search --generations 120

# Decoder neural emite candidatos a partir dos exemplos:
ouroboros generate --checkpoint validation-v2/programming/token_programmer.pt \
  --examples data/v2/tasks/novel_affine.json --output outputs-v2/my-generation --samples 16

# Executar um algoritmo realmente gerado pelo decoder:
ouroboros vm validation-v2/programming/generated/factorial.ovm --inputs 10 --trace

# Compilar a DSL imperativa para bytecode e executá-la:
ouroboros vm data/v2/examples/sum_loop.py --compile --inputs 10

# Alocar, escrever e ler heap na VM:
ouroboros vm data/v2/examples/heap.ovm
```

O JSON de especificação usa `train`, `validation` e `test`, cada qual com exemplos `{ "inputs": [...], "array": [...], "expected": número }`. Os splits devem ser separados e não contêm código da solução. A busca genética de árvores usa dois escalares; a geração de bytecode e a VM também suportam arrays e loops. **Resolver uma especificação arbitrária não é garantido.** Os relatórios preservam os candidatos inválidos e resultados incorretos.

A DSL aceita uma função `solve`, atribuições locais, operações numéricas, `if`, `while`, `for range`, `arr[i]`, `len(arr)`, `min`, `max`, `abs` e retorno. Ela é compilada para a VM; não é executada pelo interpretador Python. A VM possui combustível, limites de pilha/heap/chamadas, valores finitos e magnitude limitada. Não expõe imports, arquivos, rede ou chamadas livres. Não é um sandbox para código Python/C/ASM arbitrário.

## Componentes independentes

```bash
ouroboros bootstrap --output outputs-v2/programming --epochs 400
ouroboros communication --output outputs-v2/language --protocol factorized --steps 1500
ouroboros communication --output outputs-v2/language-free --protocol free --steps 1500
ouroboros morphology --output outputs-v2/structure
ouroboros neat --output outputs-v2/neat --generations 25
ouroboros world --language validation-v2/communication_factorized/communication.pt \
  --output outputs-v2/world --episodes 24

# Retomar o treinamento da comunicação em uma nova pasta:
ouroboros communication --protocol factorized --steps 2000 \
  --resume validation-v2/communication_factorized/communication.pt \
  --output outputs-v2/language-resumed
```

A navegação usa imitação explícita e actor-critic. O emissor/receptor é treinado antes no jogo referencial. A morfogênese segue regras de energia/estresse; ela não é organogênese espontânea. O teste estrutural aplica fases controladas de estresse e baixa energia para demonstrar o mecanismo; na execução da arena entregue o córtex permaneceu com 24 neurônios e alterou conexões, pois a regra de aumento de capacidade não foi acionada.

NEAT é um baseline de imitação de controle, com evolução estrutural real e avaliação em observações reservadas. Seu MSE não é a taxa de sucesso da arena. CfC é a implementação upstream de NCPS e participa do controlador físico. PettingZoo expõe a arena a outros algoritmos multiagente através de uma API testada.

## Arquivos e documentação

```text
ouroboros/v2/
  vm/                # Assembly, DSL, memória, controle de fluxo e execução
  programming/       # Decoder, exemplos, genética, organismo e ferramentas
  communication/     # Jogo assimétrico, protocolos e ablações
  evolution/         # CfC/morfogênese e integração NEAT
  world/             # Arena vetorizada, treino e API PettingZoo
  experiment.py      # Pipeline e critérios de integração
  dashboard.*        # Painel e console interativo
third_party/         # Fontes upstream + licenças + hashes
validation-v2/       # Pesos, programas, testes, métricas e capturas reais
configs/v2/          # Configuração do controlador NEAT
scripts/             # Instalação, execução, exportação TXT e ZIP
```

- [Arquitetura e limites de cada mecanismo](docs/architecture_v2.md)
- [Origem, commits, licenças e uso das peças do GitHub](THIRD_PARTY_NOTICES.md)
- [Relatório completo da execução](validation-v2/report.json)
- [Resultados dos testes](validation-v2/tests.txt)
- [Documentação da versão 1 preservada](docs/legacy_v1.md)

O checkpoint da comunicação inclui otimizador e RNGs e tem um teste de retomada exata em CPU. O checkpoint do mundo serve para inferência/análise e inclui as dimensões atuais do cérebro; a CLI não oferece retomada exata da simulação física. Checkpoints são carregados com `weights_only=True`.

A versão 1 continua disponível nos módulos e scripts legados; o comando principal `ouroboros` e `scripts/run_v2.py` executam a versão 2.

## TXT e ZIP

```bash
python scripts/export_source_txt.py
python scripts/package_project.py --output ../ouroboros-engine-v2.0.zip
```

`CODIGO_COMPLETO.txt` reúne fontes locais, configurações, documentação, exemplos gerados e fontes upstream com suas licenças. Pesos `.pt`, capturas PNG e assets binários acompanham o ZIP separadamente. O TXT contém código-fonte concatenado, não um binário compilado.
