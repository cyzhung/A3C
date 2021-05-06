# A3C
Use Asychronous Advantage Actor-Critic algorithm to play super mario bros
## Environment 
pytorch  1.8.1+cu102  
numpy 1.20.2  
gym  0.18.0  
gym-super-mario-bros  7.3.2  

## How to use?
Use : `python train.py --world=1  --stage=1` to train your model for 1-1  
Use : `python test.py --world=1 --stage=1` to test your model for 1-1

if you want to train model on your GPU,please modify the code  
```python
device="cpu"
```
to
```python
device="cuda"
```

in train.py (WARNING:this may cause errors in Windows OS)

## CAUTION
The default number of processes is 10  
if you want to modify this,please modify the code
```python
NUM_PROCESSES=10
```
to
```python
NUM_PROCESSES=x
```
in train.py (x is the number whatever you want)
