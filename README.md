# Rumor-detection-using-post-comments
The project uses machine learning to learn whether a given tweet is a rumor or not. We have achived validation accuracy of 79% which is on par with the paper postcom2dr (https://www.sciencedirect.com/science/article/abs/pii/S095741742101410X )

To Execute : 1. Make sure all data files/folders and the .py files are in the same folder. 2. Then use the following cmd : $ conda create -n rvt $ conda activate rvt $ conda install pip $ pip install -r requirements.txt 3. To run the py files: For data Twitter15 and Twitter 16 : python3 t15_16.py For GossipCop data : python3 gc_model.py For Politifact data :python3 pl_model.py

Files present :
                                                         1.preprocess.py : To clean the raw text data , remove stop words , stemming and lemmatization on text.
                                                                  2.encoding.py : To Embed the preprocessed data using LSTM and word2vec to get feature vectors.
                                                         3.gc_model : To build the deep learning model and train on the encoded data. After training , it save the trained model as "my_model" in the same directory.
   
