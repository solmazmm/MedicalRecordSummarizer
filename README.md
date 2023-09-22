# MedicalRecordSummarizer
Title
Medical Records Summarization, Solmaz Moradi Moghadam
Summary
In the realm of medical records management, the challenge of sifting through extensive patient histories containing diverse information can be daunting for healthcare professionals. To address this issue, I have developed a specialized summarization tool to provide summaries customized for specific medical specialties, with a particular focus on spine and brain-related issues.

Problem Statement
A medical record typically includes a multitude of documents, such as imaging reports, physician notes, progress notes, emergency department visits, and procedural details. I have developed a user-friendly summarization website that can process comprehensive PDF files containing a patient's complete medical history and generate focused summaries and highlights specific to the user specialty. In summary, this project aims to simplify the process of extracting vital information from complex medical records, benefiting both healthcare professionals and legal experts.

Project
When a user interacts with the developed Flask application, it prompts them to upload a PDF file containing the patient's entire medical record and specify the desired pages for review. The tool performs a series of tasks:
1.	Data Cleaning: The tool cleans the PDF using regular expressions to remove extraneous characters and formatting issues. This step ensures that the document is in a suitable format for analysis.
2.	Text Processing: It tokenizes the text using natural language processing (NLP) tools. It also conducts spell checks and corrects words that may have missing characters due to the quality of the scanned file.
3.	Phrase Matching: The tool employs NLP techniques for phrase matching, customizing its analysis based on the specified medical specialty (e.g., brain or spine issues). It identifies and extracts relevant phrases and keywords from the text to pinpoint important information as highlights.
4.	Summarization with BART Model: For pages containing relevant information, the tool utilizes the BART model, a transformer-based neural network, to generate concise summaries. These summaries are created to be between 50% and 75% of the length of the original texts, ensuring that essential details are retained.
5.	Output: The results are presented in a user-friendly html table. For each page containing important spine-related information, a row is generated in the output table. This row includes the page number, a summary of the content, and highlights, consist of sentences containing the relevant phrases. The "View Charts" provides visualization of the summary.
