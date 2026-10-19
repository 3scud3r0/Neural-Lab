# Evidências da versão 2.0

Esta pasta contém a execução real do perfil completo, semente 42, em CPU. `report.json` agrega os sete critérios de integração. `environment.json` registra as versões instaladas e os três commits upstream, cujos 390 arquivos originais foram conferidos por SHA-256.

## Resultados

- `programming/`: decoder autorregressivo treinado com novos exemplos por época; 11/11 famílias e 112/112 entradas finais corretas. Em 55 novos conjuntos de exemplos públicos, 53 foram interpretados corretamente. As duas falhas, em produto e máximo de array, estão registradas em `public_context_generalization`.
- `synthesis/`: expressão escalar construída por busca genética sem seed da solução, com treino, validação e teste separados.
- `user-specification/`: execução adicional de `synthesize --examples data/v2/tasks/novel_affine.json`; 4/4 testes finais corretos, sem código de referência no arquivo de exemplos.
- `neural-generation/`: execução do comando `generate` sobre um contrato de soma, com seleção por validação e avaliação posterior em teste.
- `self_modification/`: biblioteca de ferramentas em bytecode com duas revisões da soma; o programa neural substituiu a versão redundante e a chamada real retornou 8 para entradas 17 e -9. Inclui tentativas online e metabolismo numérico.
- `communication_free/`: símbolos aprendidos sem posições por atributo; 93,6% nos conceitos treinados e 20,5% nas combinações reservadas.
- `communication_factorized/`: posições por atributo declaradas e símbolos aprendidos; 100% nos 1.024 testes das combinações reservadas; 24,7% ao silenciar o canal.
- `morphology/`: adição e remoção de neurônios, redimensionamento de Parameters e religação; 12 → 15 → 18 → 15 neurônios no teste de fases metabólicas controladas.
- `neat/`: população NEAT-Python upstream com variação estrutural; controlador e avaliação de imitação em JSON.
- `world/`: 24 episódios de treino em batch 32; avaliação em 128 episódios por intervenção. Sucesso: 97/128 íntegro, 25/128 silenciado, 25/128 embaralhado, 0/128 com um único organismo cooperando.
- `tests.txt`: 41 testes aprovados, incluindo a API ParallelEnv oficial, gradientes, limites da VM e equivalência entre retomada e treino contínuo da comunicação em CPU.
- `browser-report.json` e capturas PNG: painel verificado em Chromium desktop/mobile, com execução interativa de MDC, loop compilado e interrupção por combustível.

Os snapshots são de etapas diferentes: o teste de crescimento/poda não é o mesmo cérebro da arena. Na arena desta semente o córtex permaneceu com 24 neurônios e passou por religação de conexões; a regra de crescimento não foi acionada.

## Reproduzir

Após instalar o projeto, na raiz `ouroboros-engine`:

```bash
ouroboros experiment --output outputs-v2/reproduction --profile full --require-success
python -m unittest discover -s tests -v
ouroboros synthesize --examples data/v2/tasks/novel_affine.json --output outputs-v2/specification
ouroboros generate --checkpoint validation-v2/programming/token_programmer.pt \
  --examples data/v2/tasks/sum_contract.json --output outputs-v2/generation --samples 4
ouroboros dashboard --output validation-v2 --port 8080
```

Os resultados pertencem às tarefas medidas e a uma semente de treino. Há intervalos binomiais nos relatórios; eles não medem a variância entre novos treinamentos. CUDA não foi testada porque a máquina não possui GPU disponível. Os pesos não são binários compilados de um programa; são checkpoints PyTorch carregados com `weights_only=True`.
