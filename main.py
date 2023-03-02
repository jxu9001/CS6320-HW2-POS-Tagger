# Jerry Xu
# CS 6320 Spring 2023 Homework 2 Part 1 (HMM-Based POS Tagger)
# Instructor: Professor Dan Moldovan

from HMMTagger import HMMTagger

sentence1 = 'the planet jupiter and its moons are in effect a solar system .'
sentence2 = 'computers process programs accurately .'

def main():
    # initialize the tagger
    tagger = HMMTagger()

    # load the corpus into the tagger
    tagger.load_corpus('./modified_brown')

    # initialize the initial tag probabilities, transition probabilities, and emission probabilities
    tagger.initialize_probabilities(tagger.sentences)

    # use the Viterbi algorithm to tag the two given sentences
    tag_seq1 = tagger.viterbi_decode(sentence1)
    tag_seq2 = tagger.viterbi_decode(sentence2)
    print('The most likely tag sequence for the sentence \"{}\" is {}'.format(sentence1, tag_seq1))
    print('The most likely tag sequence for the sentence \"{}\" is {}'.format(sentence2, tag_seq2))

if __name__ == '__main__':
    main()
