import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
    idf_df = pd.DataFrame(tfidf_vectorizer.idf_, index=tfidf_vectorizer.get_feature_names_out(), columns=["IDF"])

    # 4. TF-IDF Matrix
    tfidf_matrix = tfidf_vectorizer.transform(grouped_df['POITYPE'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=grouped_df['GNQXH'])

    # Save all results to an Excel file
    with pd.ExcelWriter(output_file) as writer:
        grouped_df.to_excel(writer, sheet_name='Grouped_POI', index=False)  # Merged text data
        tf_df.to_excel(writer, sheet_name='Term_Frequency')                 # Term Frequency (TF) matrix
        idf_df.to_excel(writer, sheet_name='Inverse_Document_Frequency')    # Inverse Document Frequency (IDF) matrix
        tfidf_df.to_excel(writer, sheet_name='TF_IDF_Matrix')               # TF-IDF matrix

    print(f"All intermediate results and the TF-IDF matrix have been saved to {output_file}")

# Example usage
input_file = 'spacejoin127toTFdata.xlsx'    # Input file path
output_file = 'spacejoin127TF-IDF_multiple_results.xlsx'  # Output file path
calculate_tfidf_with_intermediate_results(input_file, output_file)
