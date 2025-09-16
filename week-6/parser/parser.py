import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

""" 
    Nonterminals are generalized from below extractions from sentences (1-10.txt):
    S -> N V
    S -> N V Det N
    S -> N V Det N P N
    S -> N V P Det Adj N Conj N V
    S -> Det N V Det Adj N
    S -> N V P N
    S -> N Adv V Det N Conj N V P Det N Adv
    S -> N V Adv Det V Det N
    S -> N V Det Adj N P N Det V N P Det Adj N
    S -> N V Det Adj Adj Adj N P Det N P Det N

    S - Sentence
    NP - Noun Phrase
    VP - Verb Phrase
    ADJP - Adjective Phrase
    ADVP - Adverb Phrase
    PP - Prepositional Phrase
"""

NONTERMINALS = """
S -> S Conj S
S -> NP VP

NP -> N | Det N | Det ADJP N | NP PP | NP Conj NP
VP -> V | V NP | V PP | VP PP | VP ADVP | ADVP VP | V NP PP | V ADVP NP | VP Conj VP

ADJP -> Adj | Adj ADJP
ADVP -> Adv | Adv ADVP
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.tokenize.word_tokenize(sentence)
    # drop words without alphabetic characters and convert to lowercase
    processed = [word.lower() for word in words if any(c.isalpha() for c in word)]

    return processed


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    return [
        subtree
        for subtree in tree.subtrees(lambda t: t.label() == "NP")
        if not any(child.label() == "NP" for child in subtree.subtrees(lambda t: t != subtree))
    ]


if __name__ == "__main__":
    main()
