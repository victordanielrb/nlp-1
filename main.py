import sys
import spacy
from nltk import bigrams, FreqDist

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

TAGS = {
    "ADJ": "adjetivo",
    "ADP": "preposição",
    "ADV": "advérbio",
    "AUX": "verbo auxiliar",
    "CCONJ": "conjunção coordenativa",
    "DET": "determinante/artigo",
    "INTJ": "interjeição",
    "NOUN": "substantivo",
    "NUM": "numeral",
    "PART": "partícula",
    "PRON": "pronome",
    "PROPN": "nome próprio",
    "PUNCT": "pontuação",
    "SCONJ": "conjunção subordinativa",
    "SYM": "símbolo",
    "VERB": "verbo",
    "X": "outro",
}

NLP = spacy.load("pt_core_news_sm")

FRASES_EXEMPLO = [
    "Os alunos estudam muito.",
    "Um gato dorme no sofá.",
    "O cachorro corre rápido.",
    "Ela gosta de viajar pelo mundo.",
    "Ele toca violão muito bem.",
    "Eles caminharam juntos até a escola.",
    "O tempo está bom hoje.",
    "Meu irmão trabalha em um banco.",
    "Nós gostamos de ler livros à noite.",
    "A professora explicou a lição claramente.",
    "A a praia é bonita.",
    "O corre muito.",
]

BIGRAMAS_TREINO = FreqDist()
for frase in FRASES_EXEMPLO:
    doc = NLP(frase)
    print("tagging:", frase)
    tags_treino = [token.pos_ for token in doc]
    BIGRAMAS_TREINO.update(bigrams(tags_treino))

while True:
    frase = input("> ").strip()
    if not frase:
        break

    doc = NLP(frase)
    print("tokens:", [token.text for token in doc])
    #extrai a "tuplas" (palavra, pos, verbform) de cada token
    tags = [(token.text, token.pos_, token.morph.get("VerbForm")) for token in doc]
    print("pos tags:")
    for palavra, tag, _ in tags:
        print(f"  {palavra:<12}{tag:<6}{TAGS.get(tag, '?')}")

    erros = []
    for i in range(len(tags) - 1):
        w1, t1, _ = tags[i]
        w2, t2, vf2 = tags[i + 1]
        if t1 == "DET" and t2 == "DET":
            erros.append(f"'{w1} {w2}': dois determinantes seguidos")
        if t1 == "DET" and t2 in ("VERB", "AUX"):
            erros.append(f"'{w1} {w2}': determinante seguido de verbo")
        if t1 == "ADP" and t2 == "ADP":
            erros.append(f"'{w1} {w2}': duas preposicoes seguidas")
        if t1 == "CCONJ" and t2 == "CCONJ":
            erros.append(f"'{w1} {w2}': duas conjuncoes seguidas")
        if t1 == "ADP" and t2 == "VERB" and vf2 != ["Inf"]:
            erros.append(f"'{w1} {w2}': preposicao seguida de verbo conjugado, esperava-se infinitivo")

    novos = []
    pos_tags = [tag for _, tag, _ in tags]
    for w1, w2 in zip(pos_tags, pos_tags[1:]):
        if BIGRAMAS_TREINO[(w1, w2)] == 0:
            novos.append(f"{w1}+{w2}: bigrama de POS nunca visto no treino")

    print("[regras]", "INCORRECT" if erros else "CORRECT")
    for erro in erros:
        print(" -", erro)

    print("[dados] ", "INCORRECT" if novos else "CORRECT")
    for novo in novos:
        print(" -", novo)
