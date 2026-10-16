"""Calculadora de Probabilidade: Bayes, Binomial, Poisson, Uniforme e Exponencial.

Você digita os valores do exercício, o programa mostra o resultado e plota
automaticamente o gráfico correspondente (também salvo em ./graficos).

Dependências: pip install matplotlib numpy scipy
"""
import os
from math import comb, exp, factorial

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos")
COR = "#4C72B0"
DEST = "#DD8452"


# ------------------------------------------------------------------ cálculos
def bayes(priors, verossimilhancas, alvo):
    """P(alvo | evento) = P(evento|alvo)P(alvo) / sum(P(evento|i)P(i))."""
    total = sum(p * v for p, v in zip(priors, verossimilhancas))
    return priors[alvo] * verossimilhancas[alvo] / total, total


def binomial_pmf(n, k, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def binomial_pelo_menos(n, k, p):
    return sum(binomial_pmf(n, i, p) for i in range(k, n + 1))


def poisson_pmf(k, mu):
    return exp(-mu) * mu**k / factorial(k)


def poisson_no_maximo(k, taxa, tempo=1.0):
    mu = taxa * tempo
    return sum(poisson_pmf(i, mu) for i in range(k + 1))


def uniforme(a, b, x1, x2):
    media = (a + b) / 2
    variancia = (b - a) ** 2 / 12
    x1, x2 = max(a, x1), min(b, x2)
    prob = max(0.0, x2 - x1) / (b - a)
    return media, variancia, prob


def exponencial_maior_que(lam, t):
    return exp(-lam * t)


# ------------------------------------------------------------------ entrada
def ler(msg, tipo=float):
    return tipo(input(msg).replace(",", "."))


def ler_lista(msg):
    return [float(x.replace(",", ".")) for x in input(msg).split()]


def ler_nome(msg, padrao):
    """Pede um nome para o gráfico; Enter aceita a sugestão entre colchetes."""
    return input(f"{msg} [{padrao}]: ").strip() or padrao


def ler_nomes(msg, padroes):
    """Pede vários nomes separados por vírgula; Enter aceita as sugestões."""
    texto = input(f"{msg} [{', '.join(padroes)}]: ").strip()
    if not texto:
        return list(padroes)
    nomes = [x.strip() for x in texto.split(",") if x.strip()]
    if len(nomes) != len(padroes):
        print("  Quantidade de nomes diferente do esperado; usando as sugestões.")
        return list(padroes)
    return nomes



# ------------------------------------------------------------------ gráficos
def mostrar(nome_arquivo):
    os.makedirs(PASTA, exist_ok=True)
    caminho = os.path.join(PASTA, nome_arquivo)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    print(f"Gráfico salvo em: {caminho}")
    plt.show()
    plt.close()


def grafico_bayes(priors, veross, nomes, evento, alvo):
    total = sum(p * v for p, v in zip(priors, veross))
    post = [p * v / total for p, v in zip(priors, veross)]
    x = np.arange(len(nomes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w / 2, priors, w, label="A priori", color=COR)
    b2 = ax.bar(x + w / 2, post, w, label=f"A posteriori (dado: {evento})", color=DEST)
    r1 = ax.bar_label(b1, fmt="%.2f", padding=2)
    r2 = ax.bar_label(b2, fmt="%.2f", padding=2)
    # Destaque da hipótese alvo: borda grossa, valores em negrito e rótulo em negrito
    for barras, rotulos in ((b1, r1), (b2, r2)):
        barras[alvo].set_edgecolor("black")
        barras[alvo].set_linewidth(2.5)
        rotulos[alvo].set_fontweight("bold")
        rotulos[alvo].set_fontsize(12)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.get_xticklabels()[alvo].set_fontweight("bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Probabilidade")
    ax.set_title(f"Teorema de Bayes: P({nomes[alvo]} | {evento}) = {post[alvo]:.4f}")
    ax.legend()
    mostrar("bayes.png")


def grafico_binomial(n, k, p, r, nome):
    ks = np.arange(n + 1)
    pmf = stats.binom.pmf(ks, n, p)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(ks, pmf, color=[DEST if i >= k else COR for i in ks])
    if n <= 20:
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in pmf], fontsize=8, padding=2)
    ax.set_xlabel(f"Nº de {nome} (k)")
    ax.set_ylabel("P(X = k)")
    ax.set_title(f"Binomial(n={n}, p={p}): P(X ≥ {k}) = {r:.4f}")
    mostrar("binomial.png")


def grafico_poisson(taxa, tempo, k, r, nome):
    mu = taxa * tempo
    kmax = max(int(mu + 4 * mu**0.5) + 1, k + 3)
    ks = np.arange(kmax + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ks, stats.poisson.pmf(ks, mu), color=[DEST if i <= k else COR for i in ks])
    ax.set_xlabel(f"Nº de {nome} (k)")
    ax.set_ylabel("P(X = k)")
    ax.set_title(f"Poisson(μ={mu:g}): P(X ≤ {k}) = {r:.4f}")
    mostrar("poisson.png")


def grafico_uniforme(a, b, x1, x2, media, prob, nome, unidade):
    d = stats.uniform(loc=a, scale=b - a)
    folga = (b - a) * 0.25
    t = np.linspace(a - folga, b + folga, 600)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, d.pdf(t), color=COR, lw=2)
    ax.fill_between(t, d.pdf(t), where=(t >= x1) & (t <= x2), color=DEST, alpha=0.5,
                    label=f"P({x1:g} ≤ {nome} ≤ {x2:g}) = {prob:.4f}")
    ax.axvline(media, color="k", ls="--", label=f"Média = {media:g}{unidade}")
    ax.set_xlabel(f"{nome}{unidade}")
    ax.set_ylabel(f"f({nome})")
    ax.set_title(f"Uniforme[{a:g}, {b:g}]")
    ax.legend()
    mostrar("uniforme.png")


def grafico_exponencial(lam, x0, r, nome, unidade):
    t = np.linspace(0, max(5 / lam, x0 * 1.5), 600)
    pdf = stats.expon.pdf(t, scale=1 / lam)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, pdf, color=COR, lw=2)
    ax.fill_between(t, pdf, where=t >= x0, color=DEST, alpha=0.5,
                    label=f"P({nome} > {x0:g}) = {r:.4f}")
    ax.set_xlabel(f"{nome}{unidade}")
    ax.set_ylabel(f"f({nome})")
    ax.set_title(f"Exponencial(λ = {lam:g})")
    ax.legend()
    mostrar("exponencial.png")


# --------------------------------------------------------------------- menu
def menu():
    opcoes = {
        "1": "Teorema de Bayes",
        "2": "Binomial - P(X >= k)",
        "3": "Poisson - P(X <= k)",
        "4": "Uniforme - média, variância e P(x1 <= X <= x2)",
        "5": "Exponencial - P(T > t)",
        "0": "Sair",
    }
    while True:
        print("\n=== Calculadora de Probabilidade ===")
        for k, v in opcoes.items():
            print(f"  {k}) {v}")
        op = input("Escolha: ").strip()
        try:
            if op == "1":
                pri = ler_lista("Probabilidades a priori (separadas por espaço, ex: 0.1 0.2): ")
                ver = ler_lista("P(evento | cada hipótese) (ex: 0.01 0.02): ")
                alvo = ler(f"Índice da hipótese alvo (1 a {len(pri)}): ", int) - 1
                r, tot = bayes(pri, ver, alvo)
                print(f"P(evento) = {tot:.4f}\nP(hipótese | evento) = {r:.4f} ({r*100:.2f}%)")
                nomes = ler_nomes("Nomes das hipóteses (separados por vírgula)",
                                  [f"Hipótese {i + 1}" for i in range(len(pri))])
                evento = ler_nome("Nome do evento observado", "Evento")
                grafico_bayes(pri, ver, nomes, evento, alvo)
            elif op == "2":
                n = ler("n: ", int); k = ler("k mínimo: ", int); p = ler("p: ")
                r = binomial_pelo_menos(n, k, p)
                print(f"P(X >= {k}) = {r:.4f} ({r*100:.2f}%)")
                nome = ler_nome("Nome do que está sendo contado", "sucessos")
                grafico_binomial(n, k, p, r, nome)
            elif op == "3":
                taxa = ler("Taxa λ: "); t = ler("Intervalo de tempo: "); k = ler("k máximo: ", int)
                r = poisson_no_maximo(k, taxa, t)
                print(f"P(X <= {k}) = {r:.4f} ({r*100:.2f}%)")
                nome = ler_nome("Nome do que está sendo contado", "ocorrências")
                grafico_poisson(taxa, t, k, r, nome)
            elif op == "4":
                a = ler("a: "); b = ler("b: "); x1 = ler("x1: "); x2 = ler("x2: ")
                m, v, p = uniforme(a, b, x1, x2)
                print(f"Média = {m:.4f}\nVariância = {v:.4f}\nP = {p:.4f} ({p*100:.2f}%)")
                nome = ler_nome("Nome da variável", "X")
                un = ler_nome("Unidade (opcional, ex: ms)", "")
                grafico_uniforme(a, b, x1, x2, m, p, nome, f" ({un})" if un else "")
            elif op == "5":
                lam = ler("λ: "); t = ler("t: ")
                r = exponencial_maior_que(lam, t)
                print(f"P(T > {t}) = {r:.4f} ({r*100:.2f}%)")
                nome = ler_nome("Nome da variável", "T")
                un = ler_nome("Unidade (opcional, ex: ms)", "")
                grafico_exponencial(lam, t, r, nome, f" ({un})" if un else "")
            elif op == "0":
                break
            else:
                print("Opção inválida.")
        except (ValueError, IndexError, ZeroDivisionError) as e:
            print(f"Entrada inválida: {e}")


if __name__ == "__main__":
    menu()
