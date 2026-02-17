import bibtexparser
import re
import os

# Paths
bib_file = "/home/richard/Documents/Thesis/tex/refs.bib"
updated_bib_file = "/home/richard/Documents/Thesis/tex/refs2.bib"
latex_file = "/home/richard/Documents/Thesis/tex/main.tex"
updated_latex_file = "/home/richard/Documents/Thesis/tex/main_updated.tex"

# --- Helper functions ---
def kebab_case(text):
    text = re.sub(r'[^0-9a-zA-Z\s]', '', text)
    return '-'.join(text.lower().split())

def new_bib_key(entry):
    first_author = entry['author'].split(' and ')[0].split(',')[0].strip()
    title_words = entry['title'].split()[:2]
    return f"{kebab_case(first_author)}-{kebab_case(' '.join(title_words))}"

# --- Load BibTeX and generate key/title mappings ---
with open(bib_file, encoding='utf-8') as f:
    bib_database = bibtexparser.load(f)

key_mapping = {}
title_mapping = {}  # kebab-case title -> new ID
for entry in bib_database.entries:
    old_id = entry['ID']
    new_id = new_bib_key(entry)
    key_mapping[old_id] = new_id
    entry['ID'] = new_id
    title_mapping[kebab_case(entry['title'])] = new_id

# # Save updated BibTeX
# with open(updated_bib_file, 'w', encoding='utf-8') as f:
#     bibtexparser.dump(bib_database, f)

# print(f"Updated BibTeX saved to {updated_bib_file}")

# # Now update LaTeX file
# with open(latex_file, encoding='utf-8') as f:
#     latex_text = f.read()

# def replace_citations(match):
#     command = match.group(1)  # cite or citet
#     keys = match.group(2).split(',')
#     # Map old keys to new keys
#     new_keys = [key_mapping.get(k.strip(), k.strip()) for k in keys]
#     # Return properly formatted command with exactly one pair of braces
#     return f"\\{command}{{{','.join(new_keys)}}}"

# # Regex to match \cite{} and \citet{}
# latex_text = re.sub(r'\\(cite[t]?)\{([^}]+)\}', replace_citations, latex_text)

# with open(updated_latex_file, 'w', encoding='utf-8') as f:
#     f.write(latex_text)

# print(f"Updated LaTeX file saved to {updated_latex_file}")

# # Paths
# appendix_file = "appendix.tex"
# updated_appendix_file = "appendix_updated.tex"

# # Read appendix.tex
# with open(appendix_file, encoding='utf-8') as f:
#     appendix_text = f.read()

# # Replace citations using the same function
# appendix_text = re.sub(r'\\(cite[t]?)\{([^}]+)\}', replace_citations, appendix_text)

# # Write updated appendix
# with open(updated_appendix_file, 'w', encoding='utf-8') as f:
#     f.write(appendix_text)

# print(f"Updated appendix saved to {updated_appendix_file}")

# --- Rename PDFs ---
pdf_folder = "/home/richard/Documents/Thesis/wip_dirs/papers"

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith('.pdf'):
        name_without_ext = os.path.splitext(filename)[0].lower()
        # Clean name: remove non-alphanumeric/space
        name_cleaned = re.sub(r'[^0-9a-zA-Z\s]', '', name_without_ext)
        name_kebab = '-'.join(name_cleaned.split())
        # If it matches a title in BibTeX, rename PDF
        if name_kebab in title_mapping:
            new_pdf_name = title_mapping[name_kebab] + '.pdf'
            old_path = os.path.join(pdf_folder, filename)
            new_path = os.path.join(pdf_folder, new_pdf_name)
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' → '{new_pdf_name}'")