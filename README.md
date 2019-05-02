# Sharp Eye: Fake Photo Detector

A joint equal-contribution school project by David Wang, Subrat Mainali, Krishna Chittur and Michael Ott.

## Motivation
With systems like OpenAI’s GPT-2 on the rise, malicious agents now have a very easy time producing fake content. The need to counteract this problem is real and pressing. Thus, we believe being able to detect fake images would help us defend against such malicious actors.

## Dataset
We will be using the first phase training data-set of the [IEEE IFS-TC Image Forensics Challenge](http://ifc.recod.ic.unicamp.br/fc.website/index.py?sec=5).

## Technical Details

### First Attempt
__TODO:__
- CNN (3 channels in, ---) Layer with Relu
- Maxpool
- CNN with Relu
- Batch Normalization
- Maxpool
- Fully Connected Layer
- Batch Normalization
- Sigmoid

## Acknowledgements
Thank you to TACC and Dr. Scott Nikeum at UT Austin for giving us access to compute resources.
