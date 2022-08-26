from keras.layers import  Dense, Convolution1D, Dropout, Input, Activation, Flatten,MaxPool1D,add, AveragePooling1D, Bidirectional,GRU,LSTM,Multiply, MaxPooling1D,TimeDistributed,AvgPool1D
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.backend import sigmoid
import math
from sklearn import metrics
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras_self_attention import SeqSelfAttention
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import os,sys,re
from Bio import SeqIO
import argparse




def swish(x, beta=1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': swish})


def read_fasta(datapath):
    if os.path.exists(datapath) == False:
        print('Error: file " %s " does not exist.' % datapath)
        sys.exit(1)
    with open(datapath) as f:
        record = f.readlines()
    if re.search('>', record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % datapath)
        sys.exit(1)

    sequences = list(SeqIO.parse(datapath, "fasta"))
    sequence = []
    for i in range(len(sequences)):
        sequence.append(str(sequences[i].seq))
    return sequence

def binary(sequences):
    AA = 'ACGU'
    binary_feature = []
    for seq in sequences:
        binary = []
        for aa in seq:
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                binary.append(tag)
        binary = np.array(binary)
        # print(binary.shape)
        embedding_matrix = np.zeros((196))
        for i in range(binary.shape[0]):
            embedding_matrix[i] = binary[i]

        binary_feature.append(embedding_matrix)
    return binary_feature


# 激活函数dropout activate层
def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)
    input_at = Activation('swish')(input_bn)
    input_dp = Dropout(0.4)(input_at)
    return input_dp

# 一维卷积块
def ConvolutionBlock(input, f, k):
    A1 = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    A1 = bn_activation_dropout(A1)
    return A1

def InceptionA(input):
    A = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(B, 64, 5)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 7)
    C = ConvolutionBlock(C, 64, 7)
    return Concatenate(axis=-1)([A, B, C])

def MultiScale(input):
    A = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 3)
    D = ConvolutionBlock(input, 64, 1)
    D = ConvolutionBlock(D, 64, 5)
    D = ConvolutionBlock(D, 64, 5)
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('swish')(result)
    return result

def createModel():
    word_input = Input(shape=(49,4), name='word_input')


    overallResult1 = MultiScale(word_input)

    overallResult1 =Bidirectional(LSTM(121, return_sequences=True))(overallResult1)

    overallResult1 = Flatten()(overallResult1)
    overallResult = Dense(16, activation='sigmoid')(overallResult1)#之前是128
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)
    return Model(inputs=[word_input], outputs=[ss_output])

def Twoclassfy_evalu(y_test, y_predict):
    TP = 0                                                                                                                                                                              +0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    for i in range(len(y_test)):
        if y_predict[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 0:
            TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    Acc = (TP + TN) / (TP + FP + TN + FN)
    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_predict,pos_label=1)#poslabel正样本的标签
    auc = metrics.auc(fpr,tpr)

    return Sn, Sp, Acc, MCC,auc


def main():
    parser = argparse.ArgumentParser(description='An efficient deep learning based predictor for identifying miRNA-triggered phasiRNA loci in plant')
    parser.add_argument('--input', dest='inputpath', type=str, required=True,
                        help='query sequences to be predicted in fasta format.')
    parser.add_argument('--output', dest='outputfile', type=str, required=False,
                        help='save the prediction results in txt format.')
    args = parser.parse_args()

    inputpath = args.inputpath
    outputfile = args.outputfile
    outputfile_original = outputfile
    if outputfile_original == None:
        outputfile_original = ''
    try:
        #data collect
        t_data = inputpath
        sequences = read_fasta(t_data)

        binary1 = binary(sequences)
        binary1 = np.array(binary1)
        print(binary1.shape)
        vector = np.reshape(binary1, (binary1.shape[0], 49, 4))



        model1 = createModel()
        model1.load_weights("onehott.h5")
        predictions= model1.predict({'word_input': vector , })


        sequence = read_fasta(inputpath)
        seq = []
        for i in sequence:
            seq.append(str(i))
        probability = ['%.5f' % float(i) for i in predictions[:, 1]]
        with open(outputfile, 'w') as f:
            for i in range(int(len(x_test))):
                if float(probability[i]) > 0.5:
                    f.write(probability[i] + '*' + '\t')
                    f.write(seq[i] + '*' + '\t')
                    f.write('1' + '\n')
                else:
                    f.write(probability[i] + '*' + '\t')
                    f.write(seq[i] + '*' + '\t')
                    f.write('0' + '\n')
        print(
            'output are saved in ' + outputfile + ', and those with probability greater than 0.5 are marked with *')




    except Exception as e:
        print('Please check the format of your predicting data!')
        sys.exit(1)



if __name__ == "__main__":
    main()

