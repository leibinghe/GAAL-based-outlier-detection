# GAAL-based-outlier-detection
Two GAAL-based outlier detection models: Single-Objective Generative Adversarial Active Learning (SO-GAAL) and Multiple-Objective Generative Adversarial Active Learning (MO-GAAL).

## Environment
- Python 3.5- Tensorflow (version: 1.0.1)- Keras (version: 2.0.2)

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the parse_args function).

Run SO-GAAL:
```
python SO-GAAL.py --path Data/Annthyroid --stop_epochs 20 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9
```

Run MO-GAAL:
```
python MO-GAAL.py --path Data/Annthyroid --k 10 --stop_epochs 25 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9
```

## More Details:
Use `python SO-GAAL.py -h` or `python MO-GAAL.py -h` to get more argument setting details.

```shell
-h, --help            show this help message and exit--path [PATH]         Input data path.--k K                 Number of sub_generator.--stop_epochs STOP_EPOCHS                      Stop training generator after stop_epochs.--lr_d LR_D           Learning rate of discriminator.--lr_g LR_G           Learning rate of generator.--decay DECAY         Decay.--momentum MOMENTUM   Momentum.```

## Dataset
We provide four real-world datasets: Annthyroid, SpamBase, Waveform and WDBC in Data/

Update: September 19, 2018

