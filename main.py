import PyPDF2

import PyPDF2


def read_pdf(file_path):
    # Open the PDF file
    with open(file_path, "rb") as file:
        # Initialize a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Loop through all pages and extract text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page

    return text


# Example usage
pdf_text = read_pdf('path_to_your_pdf.pdf')
print(pdf_text[:500])  # Print the first 500 characters of the extracted text

# read words from pdfs into lists


# text processing - removing whitespace,


# chunking function with different sizes and overlap


