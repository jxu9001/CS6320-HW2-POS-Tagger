# Jerry Xu
# CS 6320 Spring 2023 Homework 2 Part 1 (HMM-Based POS Tagger)
# Instructor: Professor Dan Moldovan

from HMMTagger import HMMTagger

sentence1 = 'the planet jupiter and its moons are in effect a mini solar system .'
sentence2 = 'computers process programs accurately .'

def main():
    # initialize the tagger
    tagger = HMMTagger()

    # load the corpus into the tagger
    tagger.load_corpus('./modified_brown')

    for sentence in tagger.sentences:
        print(sentence)
    print(len(tagger.sentences))


if __name__ == '__main__':
    main()