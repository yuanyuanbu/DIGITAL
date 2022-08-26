DIGITAL: an efficient deep learning based predictor for identifying miRNA-triggered phasiRNA loci in plant.

Before running DIGITAL_Pred users shuold make sure all the following packages are installed in their Python enviroment:

     numpy == 1.19.5
     pandas == 0.22.0
     sklearn == 0.20.0
     Bio == 1.79
     keras==2.2.4
     h5py == 2.9.0
     tensorflow == 1.14
     python == 3.6



For advanced users who want to perform prediction by using their own data:

To get the information the user needs to enter for help, run:

      python DIGITAL_pred.py --help
or

      python DIGITAL_pred.py -h


Using TensorFlow backend.

usage: 
         
         DIGITAL_pred.py [-h] --input inputpath [--output OUTPUTFILE]

DIGITAL: an efficient deep learning based predictor for identifying miRNA-triggered phasiRNA loci in plant:

        -h, --help show this help message and exit

        --input inputpath query sequences to be predicted in fasta format.
 
        --output OUTPUTFILE save the prediction results.
