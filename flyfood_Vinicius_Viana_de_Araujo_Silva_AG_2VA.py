# Universidade Federal Rural de Pernambuco
# Bacharelado em Sistemas de Informação
# Estudante: Vinicius Viana de Araujo Silva

from random import randint, shuffle, random, seed
import time
import matplotlib.pyplot as plt
import numpy as np

seed(9)

tempo_inicial = time.time()


# Função responsável por retornar as coordenadas de cada ponto (linha x coluna)
def coords(matriz):
    coords = dict()

    for l in range(len(matriz)):
        for c in range(len(matriz[l])):
            if matriz[l][c] != '0':
                coords[matriz[l][c]] = [l, c]
                # ex: 'A': [2, 3]

    return coords


# Função do cálculo da distância total da rota
def calc_dist(rota, coords):
    distancia = 0
    linha_partida, coluna_partida = coords['R']

    for i in range(len(rota[0])):
        if i == 0:
            linha_atual, coluna_atual = coords[rota[0][i]]
            linha_destino, coluna_destino = coords[rota[0][i + 1]]

            distancia += abs(linha_atual - linha_partida) + abs(coluna_atual - coluna_partida) + abs(
                linha_destino - linha_atual) + abs(coluna_destino - coluna_atual)

        elif i == len(rota[0]) - 1:
            linha_atual, coluna_atual = coords[rota[0][i]]

            distancia += abs(linha_partida - linha_atual) + abs(coluna_partida - coluna_atual)
        else:
            linha_atual, coluna_atual = coords[rota[0][i]]
            linha_destino, coluna_destino = coords[rota[0][i + 1]]

            distancia += abs(linha_destino - linha_atual) + abs(coluna_destino - coluna_atual)

    return distancia


# Função de criação da população inicial
def gerar_pop_inicial(pontos, n_individuos):
    pop = []

    for n in range(n_individuos):
        shuffle(pontos)
        individuo = pontos

        if individuo not in pop:
            pop.append([individuo[:]])

    return pop


# Função de avaliação da população
def avaliar_pop(pop, coord):
    maior_dist = 0
    distancias = []

    for individuo in pop:
        dist = calc_dist(individuo, coord)
        distancias.append(dist)

        if dist > maior_dist:
            maior_dist = dist

    for d in range(len(distancias)):
        aptidao = maior_dist - distancias[d]
        pop[d].append(distancias[d])
        pop[d].append(aptidao)


# ex.: [A, B, C] --> [[A, B, C], 48, 12]]

# Função de seleção, a primeira é um torneio e a segunda é uma roleta.

def selecao(pop):
     # TORNEIO

     vencedores = []
     for _ in range(len(pop)):

         batalha = [pop[randint(0, len(pop) - 1)], pop[randint(0, len(pop) - 1)]]

         if batalha[0][2] > batalha[1][2]:
             vencedores.append(batalha[0])
         else:
             vencedores.append(batalha[1])

     return vencedores


def selecao_roleta(pop):
    # ROLETA
    vencedores = []

    soma_fitness = sum([x[2] for x in pop])
    for _ in range(len(pop)):
        roleta = randint(0, soma_fitness)
        for individuo in pop:
            roleta = roleta - individuo[2]
            if roleta <= 0:
                vencedores.append(individuo)
                break
    return vencedores


# Função para o cruzamento
def cruzar(pop):
    filhos = []
    x = 0
    j = 1
    for i in range(int(len(pop) / 2)):
        pai1 = pop[i + x][0]
        pai2 = pop[i + j][0]
        pais = [pai2, pai1]
        dividir = randint(0, len(pai1) - 1)
        cortep1 = pai1[:dividir]
        cortep2 = pai2[:dividir]

        cortes_pais = [cortep1, cortep2]

        for c in range(len(cortes_pais)):
            filho = cortes_pais[c]

            for g in pais[c]:
                if g not in filho:
                    filho.append(g)

            filhos.append([filho])
        x += 1
        j += 1
    return filhos


# Função da mutação
def mutar(pop):
    taxa = 0.03

    for individuo in pop:
        if random() <= taxa:
            individuo[0][0], individuo[0][1] = individuo[0][1], individuo[0][0]

            # [[A, B, C]] ---> [[B, C, A]]


# Função responsável por executar as gerações.
def ag(pontos, n_pop, n_geracoes, coord):
    pop = gerar_pop_inicial(pontos, n_pop)
    avaliar_pop(pop, coord)

    for g in range(n_geracoes):
        selecionados = selecao(pop)
        pop = cruzar(selecionados)
        mutar(pop)
        avaliar_pop(pop, coord)

    maior_aptidao = -1
    mais_apto = None
    dist = 0
    for individuo in pop:
        aptidao = individuo[2]

        if aptidao > maior_aptidao:
            maior_aptidao = aptidao
            mais_apto = individuo
            dist = individuo[1]

    print(' --> '.join(mais_apto[0]))
    print(f"Distância: {dist} dronômetros")

#Função main
def main():
    matriz = []
    arquivo = open('pontos_de_entrega_flyfood_2VA.txt')

    conteudo = arquivo.readlines()

    for linha in conteudo:
        if not linha.isspace():
            matriz.append(linha.replace("\n", "").split(" "))

    coordenadas = coords(matriz)
    pontos = []

    for chave in coordenadas.keys():
        if chave != 'R':
            pontos.append(chave)

    ag(pontos, 101, 100, coordenadas)


main()

tempo_final = time.time()
tempo_de_execucao = tempo_final - tempo_inicial
print(f'O tempo de execução foi {tempo_de_execucao} segundos.')

# x1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# y1 = [0.0009992122650146484, 0.0009992122650146484, 0.0009999275207519531, 0.0010008811950683594, 0.013992547988891602, 0.05496835708618164, 0.3667902946472168, 6.353364706039429, 54.365203619003296, 8290.716258764267]
# plt.plot(x1, y1)
#
#
# x2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# y2 = [0.08195376396179199,0.10693812370300293 ,0.06496286392211914 ,0.12293338775634766 ,0.11293601989746094 ,0.25185608863830566 ,0.08994913101196289 ,0.23686623573303223 ,0.24286389350891113 ,0.19688844680786133 ]
# plt.plot(x2, y2)
#
#
# plt.xlabel('Número de pontos de entregas')
# plt.ylabel('Tempo de execução em segundos')
#
# plt.title('Gráfico do tempo de execução(força bruta x genético)')
#
# plt.show()
#
#
# Percurso = np.array([[3, 0], [6, 1], [7,3 ], [5,0 ], [7,0 ],[2,0 ],[0,0],[1,1 ],[3,2 ],[2,4 ],[4,4 ],[5,4 ],[5,2 ],[7,5 ],[3,5 ],[1,7],[5,7],[4,6],[6,6 ],[7,7],[3,7],[0,6],[0,2],[0,4],[1,5],[3, 0]])
# labels = ['R', 'K', 'W', 'Z','Y', 'M','E', 'A', 'B', 'C', 'F', 'G', 'L', 'V', 'H', 'Q', 'T', 'P', 'J', 'U', 'S', 'N', 'X', 'D', 'I', 'R']
#
#
# for i in range(len(labels)):
#     label = labels[i]
#     if i < (len(labels) - 1):
#         plt.plot(
#             np.array([Percurso[i: i + 1, 0], Percurso[i: i + 1, 0]]),
#             np.array(
#                 [Percurso[i: i + 1, 1], Percurso[i + 1: i + 2, 1]]
#             ),
#             "bd-",
#             linewidth=2,
#             markersize=12,
#         )
#         plt.plot(
#             np.array(
#                 [Percurso[i: i + 1, 0], Percurso[i + 1: i + 2, 0]]
#             ),
#             np.array(
#                 [Percurso[i + 1: i + 2, 1], Percurso[i + 1: i + 2, 1]]
#             ),
#             "bd-",
#             linewidth=2,
#             markersize=12,
#         )
#         plt.annotate(
#             label,  # this is the text
#             (
#                 Percurso[i: i + 1, 0],
#                 Percurso[i: i + 1, 1],
#             ),  # these are the coordinates to position the label
#             textcoords="offset points",  # how to position the text
#             xytext=(0, 10),  # distance from text to point (x,y)
#             ha="center",
#             fontsize=12,
#         )  # horizontal alignment can be left, right or center
#     else:
#         plt.annotate(
#             label,  # this is the text
#             (
#                 Percurso[i: i + 1, 0],
#                 Percurso[i: i + 1, 1],
#             ),  # these are the coordinates to position the label
#             textcoords="offset points",  # how to position the text
#             xytext=(0, 10),  # distance from text to point (x,y)
#             ha="center",
#             fontsize=12,
#         )  # horizontal alignment can be left, right or center
# plt.title(f"Percurso com {len(labels)-2} pontos de entrega", fontsize=16)
# plt.xlabel("Coordenada X", fontsize=16)
# plt.ylabel("Coordenada Y", fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
