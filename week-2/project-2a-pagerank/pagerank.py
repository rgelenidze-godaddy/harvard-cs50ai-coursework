import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    linked_pages: set = corpus[page]

    if not linked_pages:
        # If page has no outgoing links, we choose randomly for full corpus
        total_shared_probability = 1 / len(corpus)
        return {
            page_name: total_shared_probability
            for page_name in corpus
        }
    else:
        # Transition probability is damping_factor, which is distributed equally to linked pages
        # (d / NumLinks(p))
        transition_probability: float = damping_factor / len(linked_pages)

        # Calculate additional probability gained by random jumping (dumping)
        # (1 - d) / N (distributed equally)
        jump_probability: float = (1 - damping_factor) / len(corpus)

        # Build the result distribution
        return {
            page_name:
                (transition_probability + jump_probability)
                if page_name in linked_pages  # add bonus probability to linked page
                else jump_probability  # leave only dumping probability to non-linked page
            for page_name in corpus
        }


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initial counter, counting each surf of a specific page
    counter = {page: 0 for page in corpus}

    # Generate initial_sample
    current_sample: str = random.choice(list(corpus.keys()))

    # Build markov chain (n - 1) times, 1 is already sampled initially above
    for _ in range(n - 1):
        # calculate transition probabilities with markov transition model
        transition_probabilities: dict[str, float] = transition_model(
            corpus, current_sample, damping_factor
        )

        # choose according to the probability distribution
        current_sample = random.choices(
            population=list(transition_probabilities.keys()),  # Items to choose from
            weights=list(transition_probabilities.values()),  # Distributed probability of each item
            k=1  # Number of choose
        )[0]

        # Increment counter
        counter[current_sample] += 1

    # return result statistics
    return {
        page_name: visit_count / n
        for page_name, visit_count in counter.items()
    }


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # result accumulator
    initial_prob = 1 / len(corpus)
    page_ranks = {page: initial_prob for page in corpus}

    # precalculate random choose probability (dumping), which is (1 - d) / N
    totally_random_choose_probability = (1 - damping_factor) / len(corpus)

    # consider pages without links as page linking to all the pages including itself
    corpus = {
        page_name: page_links if len(page_links) != 0 else set(corpus.keys())
        for page_name, page_links in corpus.items()
    }

    # Calculate till converge
    while True:
        # new page ranks accumulator
        new_page_ranks = {page: 0 for page in corpus}

        # Start calculating PageRank for each page
        for current_page in corpus:
            visit_probability = 0

            # get all pages which link to current_page
            connected_pages = {
                connected_page
                for connected_page in corpus
                if current_page in corpus[connected_page]
            }

            # for each page i, calculate PR(i) / NumLinks(i) [PageRank's iterative formula]
            for connected_page in connected_pages:
                # PR(i) / NumLinks(i) for i in connected pages
                visit_probability += page_ranks[connected_page] / len(corpus[connected_page])

            # Multiply by dumping factor (as sigma expression multiplies by d in formula)
            visit_probability *= damping_factor

            # final PR(p)
            pr = totally_random_choose_probability + visit_probability

            # update record
            new_page_ranks[current_page] = pr

        # get an array of each page rank change
        calculation_changes = [
            abs(old_rank - new_rank)
            for old_rank, new_rank in zip(
                page_ranks.values(), new_page_ranks.values()
            )
        ]

        # check if all changes are not more than 0.001 (converged enough to stop)
        converged_enough = all([diff <= 0.001 for diff in calculation_changes])
        if converged_enough:
            return new_page_ranks

        # shift ranks, set new as current and calculate new one until converged.
        page_ranks = new_page_ranks


if __name__ == "__main__":
    main()
