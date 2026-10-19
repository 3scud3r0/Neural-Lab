# Arquitetura executável da versão 2

## Programadora

O arquivo `programming/token_model.py` implementa uma GRU autorregressiva condicionada por exemplos numéricos. Os exemplos passam por um encoder Deep Sets, cuja agregação é invariante à ordem. O contexto entra no estado inicial, em cada entrada de token e na projeção de saída; assim o decoder pode distinguir a tarefa mesmo em prefixos longos compartilhados por programas diferentes.

O vocabulário contém opcodes e operandos, não identificadores de soluções. A máscara de geração separa posições de opcode e argumento. Saltos, tipos de índice, falta de saída e efeitos de pilha são verificados pela VM; a máscara de sintaxe não presume correção semântica.

O bootstrap supervisionado ensina famílias de programas, incluindo loops. A cada época são gerados novos conjuntos de exemplos no domínio de treino, sem tocar nos domínios reservados. O contexto usa até oito exemplos, com três escalares, o comprimento e os quatro primeiros valores do array e a saída esperada por exemplo; essa representação limita a inferência de especificações mais complexas. A validação executa o programa gerado e seleciona o checkpoint; o teste reservado é aberto depois. A busca genética constrói árvores por operadores e terminais, sem consultar a fonte da tarefa. Isso permite comparar duas formas diferentes de síntese:

- Geração neural de instruções imperativas com memória e controle de fluxo.
- Busca genética de expressões escalares em composição aberta, com profundidade limitada.

Nenhuma das duas garante resolver uma especificação arbitrária. O comando `generate` apresenta todos os candidatos e mede o selecionado em testes externos; um candidato incorreto continua registrado como incorreto.

## Máquina virtual

`vm/language.py` define instruções e validação estática. `vm/machine.py` executa uma pilha, registradores, heap alocável, array somente de leitura e frames de chamadas. `vm/compiler.py` compila uma DSL de aparência Python, com `if`, `while`, `for range`, atribuição e retorno. Não chama `eval`, `exec`, subprocessos do usuário, importação dinâmica ou funções livres.

Cada instrução consome uma unidade de combustível. O mesmo orçamento cobre chamadas e loops; ultrapassar profundidade de chamada ou limites de pilha/heap produz um resultado de falha. Valores são numéricos, finitos e com magnitude limitada. A saída tem quantidade limitada e a ausência de `OUTPUT` é falha. As mensagens de erro e traces têm comprimento limitado.

O console HTTP da VM usa o mesmo executor; ele não expõe o compilador CPython ou uma shell. Esta é uma VM de domínio limitado, não um contêiner para executar Python/C/ASM arbitrários.

## Revisão de ferramentas

Uma ferramenta é um programa de bytecode com identificador. `ToolLibrary` avalia o candidato na validação, rejeita ciclos de chamada, verifica melhoria de combustível quando solicitada e salva a revisão de forma atômica. O histórico contém o programa anterior e a identidade SHA-256 de cada revisão; `rollback()` restaura a revisão anterior.

As chamadas `CALL/RET` realmente executam essas ferramentas no mesmo modelo de pilha. O experimento inclui a troca de uma soma redundante por um programa neural correto com menos instruções executadas. Isso demonstra a alteração do repertório executável do agente e a manutenção de contratos; não significa que o agente possa modificar os testes, o intérprete ou arquivos Python do host.

## Comunicação

O jogo referencial gera conceitos combinando cor e forma. Só o emissor conhece o referente. O receptor recebe candidatos embaralhados e os símbolos emitidos; não recebe o índice correto nem a observação privada. O sucesso compartilhado é a única supervisão dos símbolos.

A variante livre tem emissor/receptor recorrentes e não fixa o significado das posições. A variante fatorada separa posições por atributo, com mapeamentos de símbolos aprendidos. Esta estrutura é um viés declarado: o sucesso composicional dela não é evidência de que uma gramática tenha surgido livremente.

Gumbel-Softmax com passagem direta discreta leva o gradiente da escolha ao emissor no treino. A avaliação usa símbolos por argmax. São comparados canal íntegro, mensagens embaralhadas, silêncio e ruído aleatório. A diagonal de combinações de atributos fica fora do treino e entra apenas na avaliação composicional. A informação mútua e o léxico são medidas empíricas, não certificados de semântica humana.

## Cooperação física

O mundo tem inércia, atrito variável, limite de velocidade, paredes, obstáculos, predador lento, custo energético e ciclo de visibilidade. O mapa estático de objetos é público; a identidade do recurso útil é privada do scout. O worker precisa decodificar uma mensagem para escolher sua chave no começo do episódio. As chaves tornam-se irreversíveis e ambas precisam apontar ao objeto correto.

Dois organismos precisam tocar e segurar o mesmo objeto útil por passos consecutivos. O recurso não pode ser colhido por um organismo sozinho. Seguir o scout depois de escolher uma chave incorreta não resolve a tarefa. Esse mecanismo mantém a informação privada necessária à decisão inicial, apesar da possibilidade de observar movimentos posteriormente.

O controlador integrado utiliza o mapa público e um canal de rádio síncrono, limitado por distância e probabilidade de perda. Os snapshots e `state()` centralizado incluem o alvo para inspeção; a observação do worker não inclui essa informação. O adaptador PettingZoo expõe mensagens, impulso, escolha e compromisso como ações separadas, permitindo outros algoritmos de treinamento.

A navegação utiliza um currículo explícito de imitação e uma atualização actor-critic. O professor usa as mesmas entradas do controlador e não revela o alvo privado ao worker: a direção recebida pelo worker decorre do receptor de mensagens. O protocolo é treinado primeiro no jogo referencial; não é um aprendizado conjunto completo desde impulsos aleatórios.

## Morfogênese e CfC

O sensor recorrente é a `CfCCell` upstream de NCPS. Um córtex adicional possui pesos de entrada, grafo recorrente com adjacência modificável, traço Hebbiano rápido e cabeças de ação/valor. O gate de espículas usa gradiente substituto. As matrizes continuam densas: não se afirma aceleração de GPU por esparsidade.

Sob estresse acumulado, `resize()` adiciona neurônios e aloca Parameters maiores. Quando a energia cai abaixo do limite, descarta neurônios de utilidade baixa e reduz as matrizes. Pesos e IDs sobreviventes são preservados. A religação remove conexões de peso baixo e adiciona conexões ausentes. O otimizador é recriado depois de redimensionar, e estados recorrentes antigos são recusados.

O experimento estrutural aplica fases explícitas de estresse e baixa energia para verificar crescimento e poda. O mundo usa as mesmas regras durante o treino, mas não precisa aumentar sua capacidade quando o controlador já resolve bem a tarefa. Nada nesses mecanismos demonstra criação espontânea de estruturas como córtex/tronco cerebral.

## NEAT upstream

Uma população real NEAT-Python evolui um controlador de navegação por especiação e mutação de nós/arestas. O fitness mede imitação de uma política de controle em observações sintéticas; o erro reservado usa uma semente independente. O relatório mostra variação estrutural da população e tamanho do vencedor. O controlador é exportado em JSON e pode ser reconstruído com `load_network()`; não utiliza pickle livre.

Esse baseline é distinto do controlador físico CfC/morfogênico. Seu MSE não é uma taxa de sucesso na arena. A entrega mantém essas métricas separadas.

## Checkpoints e reprodução

O checkpoint da comunicação inclui pesos, Adam, passo absoluto, RNG de batches, RNG Torch e RNG CUDA quando presente. Um teste compara continuação ininterrupta e retomada em CPU. Checkpoints neurais são carregados com `weights_only=True`.

O checkpoint da arena contém o cérebro com dimensões atuais, IDs, eventos estruturais, Adam e a configuração. Ele serve para inferência e análise; a CLI não oferece retomada exata do mundo. Os estados episódicos são reiniciados nas avaliações. O bootstrap da programadora salva o modelo selecionado, não uma sessão de otimização retomável.

Os perfis, sementes, versões e fontes upstream ficam registrados com cada execução. O perfil `quick` é uma passagem curta pelo pipeline, não um certificado de que todos os modelos convergiram. O perfil `full --require-success` retorna código de falha se algum critério de integração não passar.
