import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

def calculate_tfidf_with_intermediate_results(input_file, output_file):
    # Read data from the Excel file
    df = pd.read_excel(input_file, usecols=['GNQXH', 'POITYPE'])

    # 1. Merged text data
    grouped_df = df.groupby('GNQXH')['POITYPE'].apply(
        lambda x: ' '.join(x.dropna().astype(str))
    ).reset_index()

    # 2. Term Frequency (TF) calculation
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(grouped_df['POITYPE'])
    tf_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out(), index=grouped_df['GNQXH'])

    # 3. Inverse Document Frequency (IDF) calculation
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer.fit(grouped_df['POITYPE'])  # Fit only to get the IDF
    idf_values = tfidf_vectorizer.idf_  # Get the raw IDF values

    # Adjust IDF: subtract 1
    adjusted_idf_values = idf_values - 1

    # Create adjusted IDF DataFrame
    idf_df = pd.DataFrame(adjusted_idf_values, index=tfidf_vectorizer.get_feature_names_out(), columns=["Adjusted_IDF"])

    # 4. Manually compute the TF-IDF matrix using the adjusted IDF
    # Get TF values
    tf_matrix = count_matrix.toarray()

    # Apply the adjusted IDF to the TF values
    tfidf_matrix_adjusted = tf_matrix * adjusted_idf_values  # TF-IDF = TF * adjusted IDF

    # Create DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix_adjusted, columns=tfidf_vectorizer.get_feature_names_out(), index=grouped_df['GNQXH'])

    # Save all results to Excel file
    with pd.ExcelWriter(output_file) as writer:
        grouped_df.to_excel(writer, sheet_name='Grouped_POI', index=False)  # Merged text data
        tf_df.to_excel(writer, sheet_name='Term_Frequency')                 # Term Frequency (TF) matrix
        idf_df.to_excel(writer, sheet_name='Inverse_Document_Frequency')    # Adjusted Inverse Document Frequency (IDF) matrix
        tfidf_df.to_excel(writer, sheet_name='TF_IDF_Matrix')               # Adjusted TF-IDF matrix

    print(f"All intermediate results and TF-IDF matrix have been saved to {output_file}")

# Example usage
input_file = 'spacejointoTFdata.xlsx'    # Input file path
output_file = 'spacejoinTF-IDF_multiple_results(idf-1).xlsx'  # Output file path
calculate_tfidf_with_intermediate_results(input_file, output_file)
