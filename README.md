# Get inspired. LT2326 Final project

## How to run

The code can be run on mltgpu and should be run from the command line. The dataset is loaded in the data.py file from /scratch/gussuvmi/lt2326-project/. The device and number of sentences to be generated should be specified in the arguments. For example: 

``` 
python3 main.py cuda:0 10
``` 

A pretrained model is used from /scratch/gussuvmi/lt2326-project/model.pt. To retrain the model use the retrain flag:

``` 
python3 main.py cuda:0 10 --retrain
``` 
