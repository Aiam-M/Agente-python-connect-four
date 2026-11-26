# Agente ConnectX — Trabalho 2 (Engenharia de Software)

Agente para a competição [Connect X do Kaggle](https://www.kaggle.com/competitions/connectx)
vence quem alinhar 4
peças na horizontal, vertical ou diagonal. O agente final (`v3.py`) foi
submetido e obteve **score 809.3** no leaderboard (98ª posição no momento da
captura) — ver a pasta [`Provas da entrega/`](Provas%20da%20entrega/).

---

## Estrutura do repositório

| Arquivo              | Descrição                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `v1.py`              | 1ª versão: minimax (negamax) com poda alfa-beta sobre lista de listas.                                                          |
| `v2.py`              | 2ª versão: mesmo critério de avaliação, motor de busca reescrito com bitboards, avaliação incremental e tabela de transposição. |
| `v3.py`              | **Versão submetida.** v2 + melhorias de busca (killer moves, gestão de tempo) e de heurística (detecção de ameaças e paridade). |
| `test_agent.py`      | Testes automatizados (`unittest`) das três versões.                                                                             |
| `benchmark.py`       | Duelos A × B entre agentes, em paralelo, sem depender do `kaggle_environments`.                                                 |
| `teste.py`           | Validação com o `kaggle_environments` oficial (autojogo + adversários embutidos).                                               |
| `Provas da entrega/` | Capturas de tela da submissão, do leaderboard e de partidas no Kaggle.                                                          |

---

## Abordagem técnica

Todas as versões seguem a mesma interface exigida pelo Kaggle: uma função
`agent(observation, configuration)` que recebe o tabuleiro como lista plana
(`0` = vazio, `1`/`2` = jogadores) e devolve o índice da coluna a jogar, dentro
do limite de tempo por jogada (`actTimeout`, 2 s por padrão).

### Estrutura comum de decisão

1. **Atalhos táticos:** se existe vitória imediata, joga; senão, se o
   adversário vence no próximo lance, bloqueia.
2. **Busca:** _negamax_ com poda alfa-beta e **aprofundamento iterativo** —
   busca com profundidade 1, 2, 3, … até o tempo acabar, ficando sempre com o
   melhor lance da última iteração completa.
3. **Ordenação de lances:** colunas centrais primeiro (são estatisticamente
   melhores e aumentam os cortes da poda).
4. **Avaliação heurística nas folhas:** para cada "janela" de `k` células
   alinhadas, se só um jogador tem peças nela soma `10^(n-1)` (n = nº de peças)
   a favor dele; janelas mistas valem 0. Há um pequeno bônus por peças na
   coluna central. Vitórias valem `1 000 000 + profundidade` (vitórias mais
   rápidas são preferidas).

### v1 — baseline

Implementação direta em Python puro sobre uma matriz `grid[r][c]`. A avaliação
reescaneia todas as janelas a cada folha. Limite de tempo fixo de 1,2 s.

### v2 — motor de busca rápido (mesma heurística)

- **Bitboards:** cada jogador é um inteiro; a coluna `c` ocupa os bits
  `[c·H, c·H + rows)` com `H = rows + 1` (o bit extra é uma sentinela que
  impede "vazamento" entre colunas). Teste de vitória vira poucos `&` e `>>`.
- **Avaliação incremental:** cada janela guarda um código (peças do jogador 1,
  peças do jogador 2); ao jogar, só as janelas que contêm a célula são
  atualizadas, por consulta em tabela pré-calculada.
- **Tabela de transposição** (valor, profundidade, tipo de limite, melhor
  lance), reaproveitada entre iterações e entre jogadas da mesma partida.
- **Ordenação** pelo melhor lance da tabela/iteração anterior.
- Aproveitamento de iteração parcial no timeout e limite de tempo derivado de
  `configuration.actTimeout`.

### v3 — versão final

Tudo da v2, mais:

- **Killer moves:** lances que causaram corte numa mesma altura da árvore são
  tentados cedo em nós irmãos.
- **Gestão de tempo:** limite _suave_ (50 % do `actTimeout`) e _duro_ (70 %),
  usado quando a posição é instável (melhor lance mudou ou score caiu entre
  iterações). Se o banco de tempo extra (`remainingOverageTime`) está baixo,
  não usa o limite duro.
- **Primeiro lance no centro** sem busca; com uma única coluna livre, responde
  na hora.
- **Detecção de ameaças por bitboard nas folhas:** identifica vitória
  garantida no lance seguinte e ameaças duplas jogáveis do adversário, e
  bonifica ameaças com a **paridade** correta (o jogador 1 se beneficia de
  ameaças em linhas ímpares, o 2 em pares — conceito clássico do Connect 4).
- **Entradas compactas na tabela de transposição** (um inteiro por posição)
  e **coletor de lixo desligado durante a jogada**, eliminando pausas de
  > 100 ms causadas pela varredura da tabela.
- Parâmetros (`V3_PAR`, `V3_THREATS`, `V3_KILLERS`, `V3_SOFT`, `V3_HARD`)
  ajustáveis por variável de ambiente, usados apenas para ablação no
  benchmark; os valores padrão são os da submissão.

O código é genérico para qualquer `rows`, `columns` e `inarow` (há caminhos
otimizados para `k = 4`).

---

## Tecnologias utilizadas

- **Python 3.12** — o agente usa apenas a biblioteca padrão (`time`, `os`, `gc`),
  requisito para rodar no ambiente de submissão do Kaggle.
- **[kaggle-environments](https://github.com/Kaggle/kaggle-environments) 1.33.3** —
  ambiente oficial da competição, usado em `teste.py` para validar o agente.
- **unittest** (biblioteca padrão; compatível com `pytest`) — testes automatizados.
- **multiprocessing** — partidas paralelas no `benchmark.py`.
- **Git** — versionamento; **Kaggle** — submissão e avaliação.

---

## Como rodar

### 1. Ambiente

```bash
python3 -m venv trabalho2
source trabalho2/bin/activate
pip install kaggle-environments
```

> O `venv` `trabalho2/` não é versionado (ver `.gitignore`).

### 2. Testes automatizados

```bash
python -m unittest -v test_agent      # ou: pytest test_agent.py
TEST_FAST=1 python -m unittest test_agent   # versão mais rápida
```

São 29 testes, cobrindo:

- **Contrato do agente** (todas as versões): lance válido no tabuleiro vazio,
  vitória imediata nas 4 direções, bloqueio, preferência por vencer em vez de
  bloquear, nunca jogar em coluna cheia, respeito ao limite de tempo,
  tabuleiros de tamanhos não padrão, perspectiva do jogador 2 e legalidade em
  posições aleatórias.
- **Bitboards:** detecção de vitória comparada com uma implementação de
  referência e ausência de _wrap_ entre colunas.
- **Equivalência da avaliação:** a avaliação incremental da v2 produz o mesmo
  score que a avaliação completa da v1.
- **Específicos da v2/v3:** determinismo, busca mais profunda que a v1, vitória
  forçada em dois lances, `threat_cells` contra força bruta, primeiro lance no
  centro, defesa contra ameaça dupla, ablações e respeito ao limite suave.

### 3. Validação com o ambiente oficial do Kaggle

```bash
python teste.py
```

Roda o `v3.py` contra ele mesmo (como na validação da submissão — o status
esperado é `DONE` para os dois) e depois 20 partidas contra cada adversário
embutido (`random` e `negamax`), alternando quem começa.

### 4. Benchmark entre versões

```bash
python benchmark.py v3.py v2.py --pairs 20
python benchmark.py v3.py random --pairs 20
```

Cada par usa uma abertura aleatória de 4 lances jogada duas vezes, trocando
quem começa (os agentes são determinísticos, então isso dá variedade e remove
o viés de quem joga primeiro). Reporta vitórias/empates/derrotas, tempo médio
e máximo por jogada e profundidade média da busca. Opções: `--opening-plies`,
`--workers`, `--act-timeout`, `--seed`.

### 5. Submissão

Enviar o arquivo `v3.py` na aba _Submit Agent_ da competição Connect X no
Kaggle (o arquivo é autocontido e expõe `agent` e `act`).

---

## Processo de desenvolvimento

O desenvolvimento foi incremental, com cada versão validada antes de servir de
base para a seguinte:

1. **v1 — baseline funcional.** Primeiro objetivo: um agente correto e
   dentro do tempo. Minimax com alfa-beta e uma heurística de janelas simples.
   Validado com `teste.py` (autojogo e adversários embutidos).
2. **Testes como rede de segurança.** Antes de otimizar, foi escrita a
   suíte `test_agent.py`, incluindo testes de equivalência que comparam a nova
   implementação com a antiga (mesmo score de avaliação, mesma detecção de
   vitória). Isso permitiu reescrever o motor sem mudar o comportamento.
3. **v2 — desempenho.** Mantida a heurística da v1, o foco foi buscar mais
   fundo no mesmo tempo: bitboards, avaliação incremental e tabela de
   transposição. O ganho foi medido nos testes (`test_v2_searches_deeper_than_v1`)
   e em duelos.
4. **Benchmark próprio.** O `benchmark.py` foi criado para comparar versões
   de forma justa e rápida (partidas em paralelo, aberturas aleatórias com
   troca de lado, penalidade por timeout ou lance inválido).
5. **v3 — busca e heurística.** Em duas fases: (3) melhorias de busca —
   killer moves e gestão de tempo; (4) melhorias de heurística — ameaças e
   paridade. Cada recurso pode ser desligado por variável de ambiente para
   **ablação**, confirmando no benchmark que cada um contribui. Um problema de
   pausas do coletor de lixo, observado nos tempos máximos por jogada, levou à
   compactação da tabela de transposição e ao `gc.disable()` durante a busca.
6. **Submissão.** O `v3.py` foi submetido ao Kaggle, onde passou na
   validação e atingiu score 809.0.

Histórico de commits: `Agente versao 1` (v1 + validação) → `Agente v2 e v3`
(v2, v3, testes e benchmark).

---

## Resultados

- Submissão `v3.py` concluída com sucesso no Kaggle, **score 809.0**
  (posição 98 no leaderboard no momento da captura).
- Capturas da submissão, do leaderboard e do histórico de partidas em
  [`Provas da entrega/`](Provas%20da%20entrega/).
