import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, {"Harry"}, {"James"}, {"James"})
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def person_gene_count(person, one_gene, two_genes):
    """ Helper function to get gene count for a person """
    return 2 if person in two_genes else (
        1 if person in one_gene else 0
    )


def person_has_trait(person, have_trait):
    """ Helper function to get trait status for a person """
    return person in have_trait


def transfer_prob(parent_gene_count):
    if parent_gene_count == 2:
        return 1 - PROBS["mutation"]
    elif parent_gene_count == 1:
        # P(Pass) = P(Pass, !Mutate) + P(!Pass, Mutate) =
        # 0.5 * 0.99 + 0.5 * 0,01 = 0.5
        return 0.5
    else:
        return PROBS["mutation"]


def calculate_specific_number_of_gene_transfer_probability(count, fp, mp):
    # Case 1: having 0 gene is when none of the parents pass the gene:
    # P(Pass 0) = P(!FP, !MP) = P(1 - FP, 1 - MP) = (1 - FP) * (1 - MP)
    if count == 0:
        return (1 - fp) * (1 - mp)

    # Case 2: having 1 gene is when either of them (XOR) pass the gene, not both
    # P(Pass 1) = P(FP XOR MP) = P(A || B) - P (A && B) = P(A) + P(B) - 2P(A)*P(B)
    elif count == 1:
        return fp + mp - 2 * fp * mp

    # Case 3: having 2 genes is when both ob them pass the gene.
    # P(Pass 2) = P(FP, MP) = FP * MP
    else:
        return fp * mp


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    result_probability = 1  # Accumulator

    for person in people:
        gene_count = person_gene_count(person, one_gene, two_genes)
        has_trait = person_has_trait(person, have_trait)

        # Extract names of parents or None
        mother = people[person]["mother"]
        father = people[person]["father"]

        # Case 1: has parents (conditional, dependant probability)
        if mother or father:
            # get parent gene counts for a given event
            mother_gene_count = person_gene_count(mother, one_gene, two_genes)
            father_gene_count = person_gene_count(father, one_gene, two_genes)

            # get gene transfer probability from each parent, given that they have specific genes
            father_transfer_probability = transfer_prob(father_gene_count)
            mother_transfer_probability = transfer_prob(mother_gene_count)

            # calculate probability of child having N genes
            child_gene_event_probability = calculate_specific_number_of_gene_transfer_probability(
                gene_count, father_transfer_probability, mother_transfer_probability
            )

            # query probability of having trait given that child has N genes
            child_trait_event_probability = PROBS["trait"][gene_count][has_trait]

            # result child event probability is joint probability of having N genes and specific trait_status
            result_probability *= child_gene_event_probability * child_trait_event_probability
        # Case 2: Has no parents (unconditional probability), we can use PROBS data
        else:
            result_probability *= \
                PROBS["gene"][gene_count] * PROBS["trait"][gene_count][has_trait]

    return result_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # get gene count and trait status from given event
        gene_count = person_gene_count(person, one_gene, two_genes)
        has_trait = person_has_trait(person, have_trait)

        # add current event probability to the proper accumulator
        probabilities[person]["gene"][gene_count] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    # alpha * <Vector> = 1   =>   alpha = 1 / sum(for component in <Vector>)
    # so, we have to divide all component by vector sum or multiply by alpha = 1 / sum, which is the same

    for person in probabilities:
        # 1. normalize gene probabilities
        gene_distribution_sum = sum(list(probabilities[person]["gene"].values()))
        for component in probabilities[person]["gene"]:
            probabilities[person]["gene"][component] /= gene_distribution_sum

        # 2. normalize trait status probabilities
        trait_distribution_sum = sum(list(probabilities[person]["trait"].values()))
        for component in probabilities[person]["trait"]:
            probabilities[person]["trait"][component] /= trait_distribution_sum


if __name__ == "__main__":
    main()
