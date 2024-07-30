# preprocessing the data here
import numpy as np
import pandas as pd
import pysam

def load_vcf(file_path):
    # open the VCF file using pysam
    vcf_in = pysam.VariantFile(file_path)
    
    # create an empty list to store the records
    records = []
    
    # iterate over each record in the VCF file
    for record in vcf_in:
        # extract the INFO fields into a dictionary
        info_dict = {key: record.info[key] for key in record.info}
        
        # append the relevant fields to the records list
        records.append({
            'CHROM': record.contig,
            'POS': record.pos,
            'ID': record.id,
            'REF': record.ref,
            'ALT': ','.join(str(alt) for alt in record.alts),
            'QUAL': record.qual,
            'FILTER': ','.join(record.filter.keys()),
            **info_dict  # include INFO fields as columns
        })
    
    # convert the records list into a DataFrame
    data = pd.DataFrame(records)
    
    # select only numeric columns for the RBM
    numeric_data = data.select_dtypes(include=[np.number])
    
    # handle missing values by filling them with 0 (only solution I could think of)
    numeric_data = numeric_data.fillna(0)
    
    # binarize the data by setting a threshold
    threshold = numeric_data.mean()
    binary_data = (numeric_data > threshold).astype(int)
    
    # convert the DataFrame to a numpy array and return it
    return binary_data.to_numpy()