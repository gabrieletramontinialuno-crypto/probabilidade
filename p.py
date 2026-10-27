import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Configurações de estilo idênticas ao Programa.py
COR = "#4C72B0"  # Azul para a curva principal
DEST = "#DD8452"  # Laranja para a área de probabilidade destacada

PASTA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "graficos"
)


def grafico_normal(mu, sigma, x1, x2, nome="X", unidade="ms"):
    # 1. Parâmetros e cálculo da probabilidade no intervalo P(x1 <= X <= x2)
    prob = (
        stats.norm.cdf(x2, loc=mu, scale=sigma)
        - stats.norm.cdf(x1, loc=mu, scale=sigma)
    )

    # 2. Gerar pontos do eixo X (cobrindo 4 desvios padrões para cada lado)
    t = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 600)
    pdf = stats.norm.pdf(t, loc=mu, scale=sigma)

    # 3. Construção do gráfico no padrão do Programa.py
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Desenha a curva densidade de probabilidade (Gaussiana)
    ax.plot(t, pdf, color=COR, lw=2)

    # Sombra em laranja para a faixa de probabilidade [x1, x2]
    ax.fill_between(
        t,
        pdf,
        where=(t >= x1) & (t <= x2),
        color=DEST,
        alpha=0.5,
        label=f"P({x1} ≤ {nome} ≤ {x2}) = {prob:.4f}",
    )

    # Linha tracejada indicando a média
    ax.axvline(
        mu,
        color="k",
        ls="--",
        label=f"Média = {mu} ({unidade})" if unidade else f"Média = {mu}",
    )

    # Rótulos e título
    ax.set_xlabel(f"{nome} ({unidade})" if unidade else nome)
    ax.set_ylabel(f"f({nome})")

    ax.set_title(
        f"Normal(μ={mu}, σ={sigma}): "
        f"P({x1} ≤ {nome} ≤ {x2}) = {prob:.4f}"
    )

    ax.legend()

    # Salvar imagem na pasta ./graficos
    os.makedirs(PASTA, exist_ok=True)

    caminho = os.path.join(PASTA, "normal_exercicio9.png")

    plt.tight_layout()
    plt.savefig(caminho, dpi=150)

    print(
        f"Resultado: P({x1} <= X <= {x2}) = "
        f"{prob:.4f} ({prob * 100:.2f}%)"
    )
    print(f"Gráfico salvo em: {caminho}")

    plt.show()
    plt.close()


# Execução para o Exercício 9a:
# μ = 50 ms, σ = 6 ms, intervalo [44 ms, 56 ms]
if __name__ == "__main__":
    grafico_normal(
        mu=50,
        sigma=6,
        x1=44,
        x2=56,
        nome="t",
        unidade="ms",
    )
