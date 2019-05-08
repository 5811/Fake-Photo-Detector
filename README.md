# Sharp Eye: Fake Photo Detector

A joint equal-contribution school project by David Wang, Subrat Mainali, Krishna Chittur and Michael Ott.

## Motivation
With systems like OpenAI’s GPT-2 on the rise, malicious agents now have a very easy time producing fake content. The need to counteract this problem is real and pressing. Thus, we believe being able to detect fake images would help us defend against such malicious actors.

## Dataset
We will be using the first phase training data-set of the [IEEE IFS-TC Image Forensics Challenge](http://ifc.recod.ic.unicamp.br/fc.website/index.py?sec=5).

## Technical Details and Results
Check [report](https://docs.google.com/document/d/1rAeZKBjortFTPw4UEXXuXP_snXpMTjGE7pTO2fM0WoM/edit?usp=sharing).

## Acknowledgements
Thank you to TACC and Dr. Scott Nikeum at UT Austin for giving us access to compute resources.

## How to run the code
* Install pipenv for python: `pip install pipenv` or `pip3 install pipenv`.
* CD into project directory (the directory where this README is located).
* Activte environment: `pipenv shell`
* Setup: `pipenv intall`
* Download dataset from the IEEE website linked above. Structure it like so (note that the last two directories--`fake_patches` and `pristine_patches`--are empty directories.):
    * ProjectRoot
        * \training
            * \fake
                 * 231239912adbbas.png # put fake images (and there masks) here
            * \pristine
                 * 12kkfjksdfj.png # put pristine images here
            * \fake_patches
            * \pristine_patches
* Run either SimpleNet for a finetuned model by running `python src/runSimpleNet.py` or `python src/runFineTune.py`
    * If you want to control whether we fine-tune the VGG or Resnet, modify `src/FineTunedModel.py`.


