@echo off 
start /b /wait python main.py --train True --arch alexnet
start /b /wait python main.py --train True --arch Siamese
start /b /wait python main.py --train True --arch squeezenet1_1
pause