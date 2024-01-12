# Assessment of MicroRNAs Associated with Tumor Purity by Random Forest Regression
Tumor purity refers to the proportion of tumor cells in tumor tissue samples. 
This value plays an important role in understanding the mechanisms of the tumor microenvironment. 
Although various attempts have been made to predict tumor purity, attempts to predict tumor purity using miRNAs are still lacking. 
We predicted tumor purity using miRNA expression data for 16 TCGA tumor types using random forest regression. 
In addition, we identified miRNAs with high feature-importance scores and examined the extent of the change in predictive performance using informative miRNAs. 
The predictive performance obtained using only 10 miRNAs with high feature importance was close to the result obtained using all miRNAs. 
Furthermore, we also found genes targeted by miRNAs and confirmed that these genes were mainly related to immune and cancer pathways. 
Therefore, we found that the miRNA expression data could predict tumor purity well, and the results suggested the possibility that 10 miRNAs with high feature importance could be used as potential markers to predict tumor purity and to help improve our understanding of the tumor microenvironment.


<br/>

## Data

The miRNA expression data for the TCGA and PCAWG cohorts can be accessed at https://gdc.xenahubs.net. 

Tumor purity values for the samples were obtained from the consensus measurement of purity estimation (CPE) scores calculated by Aran et al. [1]

You can check all data used in the study in the data folder.


<br/>

## Download
The machine learning model and data used for training have been uploaded to this repository.


<br/>

## Citation

Dong-Yeon Nam and Je-Keun Rhee. "Assessment of MicroRNAs Associated with Tumor Purity by Random Forest Regression." Biology (Basel). 2022;11(5):787. doi:10.3390/biology11050787


<br/>

## Reference
[1] Aran, D., Sirota, M., & Butte, A. J. (2015). Systematic pan-cancer analysis of tumour purity. Nature communications, 6, 8971. https://doi.org/10.1038/ncomms9971

