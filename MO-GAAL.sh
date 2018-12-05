python MO-GAAL.py --path Data/onecluster --k 10 --stop_epochs 1500 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9

python MO-GAAL.py --path Data/Annthyroid --k 10 --stop_epochs 25 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9

python MO-GAAL.py --path Data/WDBC --k 10 --stop_epochs 200 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9

python MO-GAAL.py --path Data/Waveform --k 10 --stop_epochs 30 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9

python MO-GAAL.py --path Data/SpamBase --k 10 --stop_epochs 40 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9
