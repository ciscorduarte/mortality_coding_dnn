# dissertation

This work is a deep neural network that combines word embeddings, recurrent units, and neural attention as mechanisms for the task of automatic assignment of ICD-10 codes for causes of death by analyzing free-text descriptions in death certificates, together with the associated autopsy reports and clinical bulletins.

This neural network also explores the hierarchical nature of the input data, by building representations from the sequences of words within individual fields, which are then combined according to the sequences of fields that compose the input.

Also, a mechanism for initializing the weights of the final node of the network is exploresd, leveraging co-occurrences between classes togheter with the hierarchical structure of ICD-10.

For further information about the task, a paper on a first stage of this work was published: 
https://link.springer.com/chapter/10.1007/978-3-319-65340-2_12 

@inproceedings{Duarte2017,
author="Duarte, Francisco
and Martins, Bruno
and Pinto, C{\'a}tia Sousa
and Silva, M{\'a}rio J.",
title="A {D}eep {L}earning {M}ethod for {ICD}-10 {C}oding of {F}ree-{T}ext {D}eath {C}ertificates",
bookTitle="Proceedings of the EPIA Conference on Artificial Intelligence",
year="2017",
}

The versions used were Pyhton 3.6.0 and Keras 1.2.2
