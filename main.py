# Jerry Xu
# CS 6320 Spring 2023 Homework 2 Part 1 (HMM-Based POS Tagger)
# Instructor: Professor Dan Moldovan

from HMMTagger import HMMTagger

s1 = 'the planet jupiter and its moons are in effect a mini solar system .'
s2 = 'computers process programs accurately .'


def main():
    # initialize the tagger
    tagger = HMMTagger()

    # load the corpus into the tagger
    tagger.load_corpus('./modified_brown')

    # initialize the initial tag probabilities, transition probabilities, and emission probabilities
    tagger.initialize_probabilities(tagger.sentences)

    # use the Viterbi algorithm to tag the two given sentences
    print('The most likely tag sequence for the sentence \"{}\" is:\n {}'.format(s1, tagger.viterbi_decode(s1)))
    print('The most likely tag sequence for the sentence \"{}\" is:\n {}'.format(s2, tagger.viterbi_decode(s2)))

if __name__ == '__main__':
    main()
