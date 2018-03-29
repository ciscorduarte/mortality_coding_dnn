# ICD-10 Mortality Coding with Deep Neural Networks

This work was developed in the context of a MSc thesis at Instituto Superior Técnico, University of Lisbon.

The source code in this project leverages the keras.io deep learning libray for implementing a deep neural network that combines word embeddings, recurrent units, and neural attention as mechanisms for the task of automatic assignment of ICD-10 codes for causes of death, by analyzing free-text descriptions in death certificates, together with the associated autopsy reports and clinical bulletins.

This neural network also explores the hierarchical nature of the input data, by building representations from the sequences of words within individual fields, which are then combined according to the sequences of fields that compose the input. This part of the neural network takes it inspiration on the model advanced by Yang et al. (2016)

    @inproceedings{yang2016hierarchical,
      title={Hierarchical Attention Networks for Document Classification},
      author={Yang, Zichao and Yang, Diyi and Dyer, Chris and He, Xiaodong and Smola, Alexander J and Hovy, Eduard H},
      booktitle={Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics},
      year={2016},
      url={https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf}
    }

Moreover, a mechanism for initializing the weights of the final nodes of the network is also used, leveraging co-occurrences between classes togheter with the hierarchical structure of ICD-10.

For further information about the method, the reader can refer to the following two publications: 

    @inproceedings{duarte2017deep,
      title={A Deep Learning Method for ICD-10 Coding of Free-Text Death Certificates},
      author={Duarte, Francisco and Martins, Bruno and Pinto, C{\'a}tia Sousa and Silva, M{\'a}rio J},
      booktitle={Proceedings of the Portuguese Conference on Artificial Intelligence},
      year={2017},
      url={https://link.springer.com/chapter/10.1007/978-3-319-65340-2_12}
    }

    @article{duarte2018deep,
      title={Deep Neural Models for ICD-10 Coding of Death Certificates and Autopsy Reports in Free-Text},
      author={Duarte, Francisco and Martins, Bruno and Pinto, C{\'a}tia Sousa and Silva, M{\'a}rio J},
      journal={Journal of Biomedical Informatics},
      year={2018},
      url={http://linkinghub.elsevier.com/retrieve/pii/S1532046418300303}
    }

The code was tested with Pyhton 3.6.0 and Keras 2.1.5

### Training a model

1. Using a `.txt` file with your dataset (see `example_dataset.txt`), execute the `mortality_coding_dnn.py` indicating the dataset file directory in `line 94` of the code.

2. After the training is complete, the model saves a `.txt` file with the output (see `example_predictions.txt`)

3. The following files are saved: `modelo_full_nmf.h5`, `DICT.npy`, `FULL_CODES.npy`, `BLOCKS.npy`. These are the files needed to load the model in a different script.

4. To predict the ICD-10 code of new instances, use `predict_multi.py`. This script loads the files mentioned in the previous point and defines a `PREDICT` function. This function recieves 9 parameters: part 1 a), part 1 b), part 1 c), part 1 d), part 2, clinical bulletin diagnosis field, clinical bulletin circumnstances of admission field, clinical bulletin clinical situation field and autopsy report. See examples:

        PREDICT(['Acidente vascular cerebral isquémico do hemisfério direito'],['Estenose crítica da artéria carótida direita'],['Doença Ateroscrerótica'],[''],['Colecistite aguda gangrenada complicada com choque séptico'],[''],[''],[''],[''])
        >>> 'I632'
    
        PREDICT(['indeterminada'],[''],[''],[''],[''],[''],[''],[''],['INTOXICAÇÃO ACIDENTAL POR MONOXIDO DE CARBONO'])
        >>> 'X478'
        
        PREDICT(['Insuficiência respiratoria'],['Doença pulmonar obstrutiva crónica'],[''],[''],[''],[''],[''],[''],[''])
        >>> 'J449'
