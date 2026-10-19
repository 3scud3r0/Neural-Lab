# Código de terceiros incluído

As implementações upstream estão em `third_party/`, sem alterações nos arquivos registrados no manifesto `UPSTREAM_LOCK.json`. Cada pasta inclui sua licença e o código-fonte do pacote necessário para instalação local. Não são apenas links ou wrappers de serviços pagos.

| Projeto | Revisão incluída | Licença | Integração executada |
|---|---|---|---|
| [CodeReclaimers/neat-python](https://github.com/CodeReclaimers/neat-python) | `536985ab1d6b50973e9e9e11a0f8303a7754aecf`, tag `v1.0.0` | BSD, texto em `third_party/neat-python/LICENSE` | População, mutação estrutural, especiação e controlador exportado em JSON. |
| [mlech26l/ncps](https://github.com/mlech26l/ncps) | `695ec22d7d1831f2ced67346ef7d2f08c525bd47`, versão declarada `1.0.1` | Apache-2.0, texto em `third_party/ncps/LICENSE` | `ncps.torch.CfCCell` como sensor recorrente do cérebro morfogênico. |
| [Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | `4467a6e73ab587751942d6efc311268f094fdc5f`, tag `1.25.0` | MIT, texto em `third_party/pettingzoo/LICENSE` | `ParallelEnv`, espaços de observação/ação e teste oficial da API. |

NCPS foi fixado por commit porque a tag `v2.0.0` desse repositório identifica a implementação antiga `keras-ncp`, anterior à API CfC utilizada. A versão de distribuição do pacote selecionado é `ncps==1.0.1`; o commit e os hashes identificam exatamente as fontes entregues.

Os fontes dos pacotes foram copiados preservando copyright, licenças e nomes. Documentação original acompanha os pacotes. Os diretórios de build e metadados gerados por `pip` não fazem parte do manifesto. O comando `ouroboros doctor` verifica cada arquivo original registrado em SHA-256; ele informa divergências e impede o início do experimento integrado se a integridade falhar.

PyTorch, NumPy, Gymnasium, PyYAML e dependências transitivas são instalados pelos índices oficiais. Seus pacotes binários não estão embutidos no TXT. A API base de PettingZoo é a integração selecionada; bibliotecas opcionais de jogos Atari, ALE, MPE e outras suites não são requisitos deste laboratório.
