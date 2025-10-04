import pandas as pd
df = pd.read_csv("Multi-Labelled_Smiles_Odors_dataset.csv")
dicio = {}
smiles = df.iloc[:,0].tolist()
descritores = df.iloc[:,1].tolist()
if len(descritores) == len(smiles):
    print(len(descritores))
    print("True")
for nmr in range(len(smiles)):
    dicio[smiles[nmr]] = tuple(descritores[nmr].split(";"))
print(dicio)


#Código MANUS

import collections
from math import prod

class NaiveBayesCheiros:
    def __init__(self):
        # Dicionários para armazenar as probabilidades calculadas no treinamento
        self.prob_cheiros = collections.defaultdict(float)  # Passo 2: Probabilidades a priori
        self.prob_condicionais = collections.defaultdict(lambda: collections.defaultdict(float)) # Passo 3: Verossimilhança
        self.cheiros_possiveis = set()
        self.total_amostras = 0

    def _parse_input(self, data_dict):
        """
        Função auxiliar para converter o dicionário de entrada em uma lista
        mais fácil de manipular.
        """
        parsed_data = []
        for atomos, cheiro in data_dict.items():
            carbonos, oxigenios = map(int, atomos.split(','))
            parsed_data.append({'carbonos': carbonos, 'oxigenios': oxigenios, 'cheiro': cheiro})
        return parsed_data

    def treinar(self, dataset_dict):
        """
        Executa a fase de treinamento, calculando todas as probabilidades
        necessárias a partir do dataset.
        """
        print("--- INICIANDO FASE DE TREINAMENTO ---")

        # Converte o dicionário para um formato mais prático
        dados_treino = self._parse_input(dataset_dict)
        self.total_amostras = len(dados_treino)
        self.cheiros_possiveis = {d['cheiro'] for d in dados_treino}

        # --- PASSO 2: Calcular as Probabilidades a Priori dos Cheiros ---
        print("\n[PASSO 2] Calculando probabilidades a priori de cada cheiro...")
        contagem_cheiros = collections.Counter(d['cheiro'] for d in dados_treino)
        
        for cheiro, contagem in contagem_cheiros.items():
            self.prob_cheiros[cheiro] = contagem / self.total_amostras
            print(f"  P({cheiro}) = {contagem}/{self.total_amostras} = {self.prob_cheiros[cheiro]:.2f}")

        # --- PASSO 3: Calcular as Probabilidades Condicionais (Verossimilhança) ---
        print("\n[PASSO 3] Calculando probabilidades condicionais (verossimilhança)...")
        
        for cheiro in self.cheiros_possiveis:
            # Filtra o dataset para conter apenas amostras do cheiro atual
            amostras_do_cheiro = [d for d in dados_treino if d['cheiro'] == cheiro]
            total_no_cheiro = len(amostras_do_cheiro)

            # Calcula a probabilidade para o número de carbonos
            contagem_carbonos = collections.Counter(d['carbonos'] for d in amostras_do_cheiro)
            for valor, contagem in contagem_carbonos.items():
                # Chave: (feature, valor, cheiro) -> Ex: ('carbonos', 6, 'Frutado')
                chave = ('carbonos', valor, cheiro)
                # Usamos suavização de Laplace (adicionar 1) para evitar probabilidade zero
                # se um valor nunca aparecer para um cheiro.
                self.prob_condicionais[chave] = (contagem + 1) / (total_no_cheiro + len(contagem_carbonos))
                # print(f"  P(Carbonos={valor} | {cheiro}) = {contagem}/{total_no_cheiro}")

            # Calcula a probabilidade para o número de oxigênios
            contagem_oxigenios = collections.Counter(d['oxigenios'] for d in amostras_do_cheiro)
            for valor, contagem in contagem_oxigenios.items():
                chave = ('oxigenios', valor, cheiro)
                self.prob_condicionais[chave] = (contagem + 1) / (total_no_cheiro + len(contagem_oxigenios))
                # print(f"  P(Oxigênios={valor} | {cheiro}) = {contagem}/{total_no_cheiro}")
        
        print("  Probabilidades condicionais calculadas com sucesso (com suavização de Laplace).")
        print("\n--- TREINAMENTO CONCLUÍDO ---")


    def prever(self, num_carbonos, num_oxigenios):
        """
        Executa a fase de predição para uma nova molécula.
        """
        print(f"\n--- INICIANDO PREDIÇÃO para Carbonos={num_carbonos}, Oxigênios={num_oxigenios} ---")
        
        pontuacoes = {}

        # --- PASSO 4: Calcular a Pontuação Final para cada Cheiro ---
        print("\n[PASSO 4] Calculando a pontuação para cada cheiro possível...")
        for cheiro in self.cheiros_possiveis:
            # 1. Começa com a probabilidade a priori do cheiro
            prob_a_priori = self.prob_cheiros[cheiro]
            
            # 2. Busca as probabilidades condicionais
            # Se um valor nunca foi visto para um cheiro, a probabilidade será muito baixa (devido à suavização), mas não zero.
            prob_carbono_dado_cheiro = self.prob_condicionais.get(('carbonos', num_carbonos, cheiro), 1e-6) # Valor pequeno para evitar zero
            prob_oxigenio_dado_cheiro = self.prob_condicionais.get(('oxigenios', num_oxigenios, cheiro), 1e-6)

            # 3. Calcula a pontuação final (produto das probabilidades)
            # Usamos a função prod() para multiplicar todos os valores em uma lista
            pontuacao_final = prod([
                prob_a_priori,
                prob_carbono_dado_cheiro,
                prob_oxigenio_dado_cheiro
            ])
            
            pontuacoes[cheiro] = pontuacao_final
            print(f"  Pontuação para '{cheiro}': {pontuacao_final:.6f}")

        # --- PASSO 5: A Decisão Final ---
        print("\n[PASSO 5] Encontrando o cheiro com a maior pontuação...")
        if not pontuacoes:
            return "Nenhum cheiro previsto. O modelo foi treinado?", 0.0

        cheiro_previsto = max(pontuacoes, key=pontuacoes.get)
        maior_pontuacao = pontuacoes[cheiro_previsto]
        
        print(f"  O cheiro com a maior pontuação é '{cheiro_previsto}'.")
        
        return cheiro_previsto, maior_pontuacao

# --- Exemplo de Uso ---

# 1. Criar uma instância do classificador
modelo_nb = NaiveBayesCheiros()

# 2. Treinar o modelo com o nosso dataset
modelo_nb.treinar(dataset)

# 3. Fazer uma predição para uma nova molécula
# Exemplo: uma molécula com 7 carbonos e 2 oxigênios
carbonos_novo = 7
oxigenios_novo = 2
cheiro_final, pontuacao = modelo_nb.prever(carbonos_novo, oxigenios_novo)

print("\n--- RESULTADO FINAL ---")
print(f"A predição de cheiro para uma molécula com {carbonos_novo} carbonos e {oxigenios_novo} oxigênios é: **{cheiro_final}** (Pontuação: {pontuacao:.6f})")

# Outro exemplo: uma molécula com 5 carbonos e 1 oxigênio
carbonos_novo_2 = 5
oxigenios_novo_2 = 1
cheiro_final_2, pontuacao_2 = modelo_nb.prever(carbonos_novo_2, oxigenios_novo_2)

print("\n--- RESULTADO FINAL ---")
print(f"A predição de cheiro para uma molécula com {carbonos_novo_2} carbonos e {oxigenios_novo_2} oxigênios é: **{cheiro_final_2}** (Pontuação: {pontuacao_2:.6f})")


