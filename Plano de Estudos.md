# Roteiro: Do Zero às PINNs em Cosmologia

---

## Como usar este documento

Este roteiro acompanha seu aprendizado de machine learning até Physics-Informed Neural Networks (PINNs) aplicadas à cosmologia. Ele tem quatro fases progressivas, cada uma com um mini-projeto de consolidação, e culmina em um projeto final com potencial acadêmico.

**Informe ao orientador em qual fase e etapa você está no início de cada conversa.**

---

## Fase 0 — Fundamentos de ML sem framework (2–4 semanas)

**Objetivo:** entender o que acontece por baixo antes de usar abstrações. Tudo em NumPy puro.

### Etapas

**0.1 — Gradiente descendente à mão**
- Implemente regressão linear com MSE loss
- Calcule o gradiente analiticamente e faça o update manual
- Experimente diferentes learning rates e observe o comportamento
- Conceitos-chave: loss surface, convergência, hiperparâmetros

**0.2 — MLP do zero**
- Implemente camadas densas como operações matriciais
- Implemente ativações: ReLU, sigmoid, tanh — e suas derivadas
- Forward pass: entenda o fluxo de dados
- Conceitos-chave: representação em camadas, não-linearidade

**0.3 — Backpropagation manual**
- Derive e implemente backprop para uma rede de 2 camadas
- Verifique seus gradientes com gradient checking numérico
- Treine em um dataset sintético e confirme que a loss cai
- Conceitos-chave: regra da cadeia, grafo computacional implícito

**0.4 — Autograd minimalista**
- Implemente um grafo computacional onde cada operação registra seus inputs
- Implemente `.backward()` que propaga gradientes pelo grafo
- Referência: `micrograd` do Karpathy (leia o código inteiro — são ~150 linhas)
- Conceitos-chave: grafo acíclico dirigido, acumulação de gradientes

### Mini-projeto 0 — *Ajuste de curva de luz*

**Contexto:** Dado um conjunto de pontos (t, flux) de uma curva de luz sintética de supernova Ia, ajuste um modelo paramétrico usando sua MLP do zero (sem PyTorch).

**Entregáveis:**
- MLP com backprop manual que minimize o resíduo dos dados
- Plot: dados observados vs. curva ajustada
- Análise: como a profundidade e largura da rede afetam o ajuste?

**Por que este projeto:** conecta diretamente com sua área observacional e força você a depurar backprop em um problema real — sem a rede de segurança dos frameworks.

---

## Fase 1 — PyTorch e EDOs (2–3 semanas)

**Objetivo:** dominar o framework que usará em PINNs e conectar gradientes automáticos com equações diferenciais.

### Etapas

**1.1 — PyTorch básico**
- Refaça a MLP da Fase 0 em PyTorch
- Entenda: `Tensor`, `requires_grad`, `autograd`, `optimizer.step()`
- Compare: onde o framework faz o que você fez à mão?
- Conceitos-chave: grafo computacional dinâmico, `backward()` do PyTorch

**1.2 — `torch.autograd.grad` vs `.backward()`**
- Entenda a diferença: `.backward()` acumula gradientes nos `.grad`; `autograd.grad` retorna gradientes explicitamente
- Exercício: dado `y = f(x)`, calcule `dy/dx` e `d²y/dx²` usando `autograd.grad`
- Isso é o coração das PINNs: derivadas da rede em relação à entrada, não aos parâmetros
- Conceitos-chave: `create_graph=True`, segunda ordem, Jacobians

**1.3 — Aproximação de funções**
- Treine uma rede para aprender `f(x) = sin(3x) · e^(-x/2)` em `[0, 4π]`
- Experimente: tamanho da rede, número de pontos de treino, overfitting
- Conceitos-chave: capacidade da rede, generalização, regularização

**1.4 — Proto-PINN: EDO sem chamar de PINN**
- EDO alvo: `u' + u = 0`, `u(0) = 1`, solução: `e^(-x)`
- Represente `u(x)` como uma rede neural
- Minimize: `|u'(x) + u(x)|²` usando `autograd.grad` para calcular `u'`
- Adicione a condição inicial como termo na loss
- Conceitos-chave: rede como ansatz, residual de EDO, collocation points

### Mini-projeto 1 — *Oscilador anarmônico interativo*

**Contexto:** lúdico, mas com física real. Você tem um oscilador com potencial `V(x) = x²/2 + εx⁴/4`. Resolva numericamente com `scipy.solve_ivp` e depois treine uma rede PyTorch para aprender a trajetória `x(t)`.

**Entregáveis:**
- Solução numérica de referência
- Rede PyTorch treinada para aproximar `x(t)` — sem PINN ainda, só dados
- Plot interativo (matplotlib com slider): varie `ε` e compare solução vs. rede
- Análise: a rede generaliza para `ε` fora do treinamento?

**Por que este projeto:** força você a usar `autograd.grad` para análise, trabalhar com séries temporais, e pensar sobre generalização — tudo antes das PINNs formais.

---

## Fase 2 — PINNs propriamente ditas (3–5 semanas)

**Objetivo:** dominar a arquitetura, a loss composta e as sutilezas de treinamento.

### A estrutura geral de uma PINN

```
L_total = λ_r · L_residual + λ_bc · L_contorno + λ_data · L_dados
```

Onde:
- `L_residual`: quão bem a rede satisfaz a EDO/EDP nos pontos de collocation
- `L_contorno`: quão bem satisfaz condições de contorno/iniciais
- `L_dados`: quão bem ajusta dados observados (quando existem)
- `λ_i`: pesos que balanceam os termos (hiperparâmetro crucial)

### Etapas

**2.1 — Equação do calor 1D**
- `∂u/∂t = α ∂²u/∂x²`, domínio `[0,1] × [0,1]`
- Condição inicial: `u(x,0) = sin(πx)`
- Condições de contorno: `u(0,t) = u(1,t) = 0`
- Solução analítica: `u = sin(πx) · e^(-α π² t)`
- Meça o erro L² em relação à solução analítica

**2.2 — Equação de Burgers**
- `∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²`
- Não-linear, forma descontinuidades (choques) para ν pequeno
- Benchmark clássico da literatura de PINNs
- Questão central: onde a PINN falha? Por quê?

**2.3 — Oscilador harmônico amortecido**
- `u'' + 2γu' + ω²u = 0`, `u(0) = 1`, `u'(0) = 0`
- Três regimes: subamortecido, criticamente amortecido, superamortecido
- Compare a PINN com `scipy` nos três regimes
- Questão central: a PINN captura a transição entre regimes?

**2.4 — Tópicos de treinamento** (estudar em paralelo aos projetos)
- Amostragem de collocation: aleatória vs. grid vs. Latin hypercube
- Balanceamento da loss: por que `λ_bc >> λ_r` frequentemente funciona melhor?
- *Spectral bias*: por que PINNs falham em alta frequência? Leia Rahaman et al. 2019
- Estratégias de curriculum: treinar com poucas restrições primeiro

### Mini-projeto 2 — *Dungeon físico*

**Contexto:** lúdico. Um agente em um corredor 1D obedece a equação de movimento com atrito. Você não tem acesso às equações — só a trajetórias observadas. Use uma PINN para descobrir o coeficiente de atrito γ (problema inverso).

**Entregáveis:**
- Geração de trajetórias sintéticas com parâmetros conhecidos
- PINN que trata γ como parâmetro treinável (não fixo)
- Recuperação de γ a partir de dados ruidosos
- Análise: como o nível de ruído afeta a recuperação?

**Por que este projeto:** o problema inverso é onde as PINNs brilham — e é exatamente o que o projeto final usará em escala cosmológica.

---

## Fase 3 — Tópicos avançados (contínuo, em paralelo à Fase 4)

**Objetivo:** entender por que PINNs falham e como consertá-las. A literatura aqui ainda está aberta.

### Tópicos

**3.1 — Adaptive sampling**
- Residual Adaptive Refinement (RAR): mover pontos para onde `L_residual` é maior
- Implement um ciclo de retreinamento com redistribuição de pontos
- Referência: Lu et al. 2021 (DeepXDE paper)

**3.2 — Fourier feature networks**
- Camada de embedding: `γ(x) = [sin(Bx), cos(Bx)]` antes da MLP
- Mitiga spectral bias em problemas multi-escala
- Referência: Tancik et al. 2020

**3.3 — Arquiteturas alternativas**
- Modified MLP: conexões skip + normalização adaptativa
- ResNets dentro da PINN
- Quando cada arquitetura ajuda?

**3.4 — Análise de falhas**
- Leia: Krishnapriyan et al. 2021 — "Characterizing possible failure modes in PINNs"
- Reproduza um dos exemplos de falha do paper
- Questão central: você consegue identificar o regime de falha antes de treinar?

---

## Fase 4 — PINNs em cosmologia (projeto final)

**Objetivo:** aplicar o toolkit completo a equações cosmológicas reais. Progressão de dificuldade crescente; o item 4.3 é o projeto final.

### Etapas preparatórias

**4.1 — Background cosmológico com PINN**
- Sistema de Friedmann: `H² = (8πG/3)ρ`, `ρ' = -3H(ρ + p)`
- PINN que resolve `H(z)` dado `Ω_m, Ω_Λ`
- Compare com integração numérica direta
- Questão: a PINN é mais rápida? Em que regime?

**4.2 — Problema inverso cosmológico simples**
- Dado `H(z)` "observado" (sintético, com ruído), recupere `Ω_m`
- Trate `Ω_m` como parâmetro treinável
- Compare com MCMC clássico (use `emcee`)
- Questão: onde cada método é superior?

---

## Projeto Final — *Reconstrução de potencial inflacionário via PINN*

### Motivação

A equação de Mukhanov-Sasaki governa as perturbações escalares durante a inflação:

```
u_k'' + (k² - z''/z) u_k = 0
```

onde `z = aφ'/H` e as condições de Bunch-Davies no passado distante determinam o estado de vácuo. O espectro de potência primordial `P(k)` é determinado pelo comportamento assintótico de `u_k`.

O problema inverso: **dado um espectro `P(k)` observado (e.g., desvios do espectro de Harrison-Zel'dovich), reconstruir o potencial inflacionário `V(φ)`.**

### Estrutura do projeto

**Etapa A — Forward problem**
- Implemente a equação de Mukhanov-Sasaki como PINN
- Valide contra soluções analíticas (slow-roll de ordem zero)
- Calcule `P(k)` a partir da solução e compare com CAMB

**Etapa B — Geração de dados sintéticos**
- Escolha um modelo inflacionário (ex: Starobinsky, natural inflation)
- Use CAMB/CLASS para gerar `P(k)` com features controladas
- Adicione ruído compatível com a sensibilidade do Planck

**Etapa C — Problema inverso**
- Trate `V(φ)` como uma rede neural auxiliar (dentro da PINN principal)
- Minimize o resíduo da equação de M-S + discrepância com `P(k)` observado
- Regularização: prior suave em `V(φ)` (ex: penalizar oscilações)

**Etapa D — Análise e interpretação**
- O `V(φ)` recuperado é fisicamente razoável?
- Como o resultado depende do nível de ruído e da cobertura em `k`?
- Compare com reconstruções por outros métodos (ex: reconstrução espectral)

### Conexão acadêmica

Este projeto tem estrutura de paper. A contribuição original seria:
1. Aplicação de PINNs ao problema inverso inflacionário (ainda pouco explorado)
2. Comparação com métodos existentes de reconstrução de V(φ)
3. Análise de degenerescências (subdeterminação empírica — conecta com seu interesse em filosofia da ciência)

**Referências de partida:**
- Raissi, Perdikaris, Karniadakis 2019 — paper original de PINNs
- Chung & Shiu 2022 — ML para cosmologia inflacionária
- Guo et al. 2022 — reconstrução de potencial por ML
- CAMB/CLASS documentation

---

## Recursos por fase

| Fase | Recurso principal | Complementar |
|------|-------------------|--------------|
| 0 | `micrograd` — Karpathy (GitHub) | 3Blue1Brown — Neural Networks playlist |
| 1 | PyTorch docs — autograd | Fast.ai Part 1 (caps. 1–4) |
| 2 | Raissi et al. 2019 | `DeepXDE` source code |
| 3 | Krishnapriyan et al. 2021 | Tancik et al. 2020 (Fourier features) |
| 4 | Lu et al. 2021 (DeepXDE) | CAMB documentation |

---

# Instruções para o Project do Claude

> **Copie o texto abaixo para o campo "Instruções" ao criar seu Project.**

---

Você é um orientador de estudos dirigidos especializado em machine learning e física computacional. O estudante é Nícolas, mestrando em cosmologia observacional no Brasil, com Python intermediário (confortável com NumPy/matplotlib, menos experiente com orientação a objetos), estudando em ritmo flexível paralelo ao mestrado.

**Roteiro de referência:** Nícolas segue um roteiro estruturado em 5 fases (0–4) que vai de fundamentos de ML até Physics-Informed Neural Networks (PINNs) aplicadas à cosmologia inflacionária. O roteiro completo está no arquivo de conhecimento deste Project. No início de cada conversa, ele informará em qual fase e etapa está.

## Princípios de conduta

**1. Seja socrático por padrão.**
Não forneça soluções ou respostas diretas a menos que Nícolas peça explicitamente (com frases como "me dá a resposta", "pode resolver", "me mostra o código completo"). Antes disso, responda com perguntas que guiem o raciocínio:
- "O que você esperaria que acontecesse com o gradiente se...?"
- "Você consegue rastrear o valor de x nesse ponto do código?"
- "Essa equação te lembra alguma coisa da mecânica que você já viu?"

**2. Bugs têm tratamento diferenciado.**
- *Bug conceitual* (lógica errada, entendimento equivocado de como algo funciona): socrático. Aponte onde o raciocínio diverge e faça perguntas.
- *Bug de sintaxe* (erro de Python, shape mismatch, import faltando): direto. Indique o problema e a correção sem rodeios.
- Em caso de dúvida sobre o tipo, pergunte: "Isso é um problema de entendimento ou só de sintaxe?"

**3. Expanda sem resolver.**
Quando Nícolas pedir para expandir uma explicação, aprofunde o conceito com exemplos, analogias com física que ele já conhece (mecânica quântica, perturbações cosmológicas, MCMC) e perguntas que testem a compreensão — mas não entregue o passo seguinte do roteiro sem ele ter trabalhado o atual.

**4. Calibre pelo nível.**
Python intermediário: não explique sintaxe básica, mas explique padrões como decorators, classes para módulos de rede, broadcasting não-óbvio. Física: assuma familiaridade com equações de campo, perturbações lineares e inferência bayesiana básica.

**5. Seja direto sobre o roteiro.**
Se Nícolas perguntar "estou pronto para avançar?", avalie com base no que ele demonstrou na conversa — não apenas no que ele diz. Se há lacunas evidentes, aponte-as antes de confirmar a progressão.

## Frases de gatilho

- "pode resolver" / "me dá a resposta" / "mostra o código" → responda diretamente
- "expande" / "aprofunda" / "não entendi" → socrático + analogias
- "to travado há muito tempo" → dê uma dica direta, não a solução
- "fase X, etapa Y" → contextualize suas respostas para aquela etapa específica do roteiro

## Tom

Rigoroso mas encorajador. Não elogie de forma vaga ("ótimo!") — quando algo estiver correto, diga *por que* está correto. Quando algo estiver errado, trate como oportunidade de aprendizado, não como falha.
