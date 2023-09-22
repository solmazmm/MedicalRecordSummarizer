from flask import Flask, request, render_template, session
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import tempfile
from spacy.matcher import PhraseMatcher
import io
import os
import spacy
import re
from collections import Counter
from transformers import BartForConditionalGeneration, BartTokenizer
from spellchecker import SpellChecker
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib

matplotlib.use('Agg')


def generate_html_table(summary_dict, highlights_dict):
    # Initialize an empty string to store the HTML table content
    html_table = ""
    html_table += "<table>"
    html_table += "<tr><th>Page Number</th><th>Summary of the Page</th><th>Highlights</th></tr>"
    # Iterate over page numbers
    for page_number in summary_dict.keys():
        summary_text = summary_dict.get(page_number, "")
        highlight_text = highlights_dict.get(page_number, "")
        html_table += f"<tr><td>{page_number}</td><td>{summary_text}</td><td>{highlight_text}</td></tr>"
    html_table += "</table>"
    return html_table


def extractive_summary(data):
    if data is not None:
        model_name = "facebook/bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        original_words = tokenizer.tokenize(data)  # Tokenize the input text
        num_original_words = len(original_words)  # Count the tokens as words

        length_percentage = 75  # can be adjusted
        max_length = int(num_original_words * (length_percentage / 100))
        min_length = int(max_length * 0.5)  # can be adjusted

        # Generating the summary
        input_ids = tokenizer.encode(data, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return generated_summary
    else:
        return None


def importants(data):
    mylist = []

    nlp = spacy.load("en_core_web_sm")

    phrase_matcher = PhraseMatcher(nlp.vocab)
    phrases = ['spine', 'Spine', 'SPINE', 'skull', 'Skull', 'SKULL', 'Brain', 'brain', 'BRAIN',
               'cervical', 'Cervical', 'CERVICAL', 'lumbar', 'Lumbar', 'LUMBAR', 'thorasic', 'Thorasic',
               'THORASIC', 'vertebral', 'Vertebral', 'VERTEBRAL', 'disc', 'DISC', 'Disc', 'low back', 'LOW BACK',
               'Low Back', 'Low back', 'Dorsal', 'DORSAL', 'dorsal', 'concussion', 'Concussion', 'CONCUSSION',
               'conscious', 'Headache', 'headache', 'HEADACHE'
                                                    'Conscious', 'CONSCIOUS', 'foreheaed', 'Forehead', 'FOREHEAD',
               'Neck', 'neck', 'NECK']
    patterns = [nlp(data) for data in phrases]
    phrase_matcher.add('SS', None, *patterns)

    doc = nlp(data)

    sett = set()
    for sent in doc.sents:
        for match_id, start, end in phrase_matcher(nlp(sent.text)):
            if nlp.vocab.strings[match_id] in ["SS"]:
                sett.add(sent.text)
    mylist = list(sett)
    return mylist


app = Flask(__name__)
app.secret_key = 's0lmazs0lmaz'


def processing(file_path, page_no, maxpages):
    ResultSentences = dict()
    ResultSummary = dict()

    with open(file_path, 'rb') as fp:  # Open the PDF file
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        for pageNumber, page in enumerate(PDFPage.get_pages(fp)):
            if page_no <= pageNumber <= maxpages and pageNumber != 20:
                interpreter.process_page(page)

                data = retstr.getvalue()
                data = data.replace('\xad', '')
                data = data.replace('\N{SOFT HYPHEN}', '')
                data = re.sub(r'[^a-zA-Z0-9\s\-/]', '', data)
                data = re.sub(r'\s+', ' ', data)
                # replacing common line break patterns with a newline character
                data = re.sub(r'\s*[\r\n]+\s*', '\n', data)
                # ++++++++++ Spell check
                spell = SpellChecker()
                words = word_tokenize(data)
                corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in
                                   words]
                # ++++++++++ clean up non-english words
                stop_words = set(stopwords.words('english'))
                filtered_words = [word for word in corrected_words if word.lower() not in stop_words]
                data = ' '.join(filtered_words)
                im = importants(data)
                if im:
                    ResultSentences[pageNumber] = im
                    summ = extractive_summary(data)
                    if summ is not None:
                        ResultSummary[pageNumber] = summ
                data = ''
                retstr.truncate(0)
                retstr.seek(0)
    return ResultSummary, ResultSentences


def generate_chart_data(output_highlights, n=10):
    # Combine all sentences into a single list
    all_sentences = [sentence for sentences in output_highlights.values() for sentence in sentences]
    # Tokenize the sentences into words
    all_words = [word for sentence in all_sentences for word in sentence.split()]
    # +++++
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in all_words if word.lower() not in stop_words]
    # +++++
    # Calculate word frequencies
    word_frequencies = Counter(filtered_words)
    # top N most frequent words
    top_words = word_frequencies.most_common(n)
    # if top_words is empty or has fewer than two elements:
    if not top_words or len(top_words) < 2:
        return {
            'bar_labels': [],
            'bar_values': [],
            'word_cloud_image_base64': '',
        }
    bar_labels, bar_values = zip(*top_words)

    # ++++++++++++++++++++++++++++++++

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)

    # Plot the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the chart as PNG image
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    plt.close()

    # dictionary containing data for both the Bar Chart and Word Cloud
    chart_data = {
        'bar_labels': bar_labels,
        'bar_values': bar_values,
        'word_cloud_image_base64': image_base64
    }
    return chart_data


@app.route('/chart')
def chart():
    output_highlights = session.get('output_highlights')

    if not output_highlights:
        return "Data not found in the session."
    chart_data = generate_chart_data(output_highlights, n=10)  # Adjust the number as needed

    return render_template('chart2.html', chart_data=chart_data)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index2.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index2.html', error='No selected file')

        try:
            # Get user input values for "from page" and "to page"
            page_no = int(request.form['page_no'])
            maxpages = int(request.form['maxpages'])

            if file:
                try:
                    # a temporary directory to store the uploaded file
                    temp_dir = tempfile.mkdtemp()
                    file_path = os.path.join(temp_dir, file.filename)
                    file.save(file_path)

                    # Call the processing function with the file path, page_no, and maxpages
                    output_summary, output_highlights = processing(file_path, page_no, maxpages)
                    session['output_highlights'] = output_highlights
                    # Clean up temporaries
                    os.remove(file_path)
                    os.rmdir(temp_dir)

                    # pass both dictionaries to generate the HTML table
                    html_table = generate_html_table(output_summary, output_highlights)
                    return render_template('summary.html', html_table=html_table)

                except Exception as e:
                    return render_template('index2.html', error=f'Error processing PDF: {str(e)}')

        except ValueError:
            return render_template('index2.html', error='From page and To page must be valid numbers.')

    return render_template('index2.html')


if __name__ == '__main__':
    app.run(debug=True)
