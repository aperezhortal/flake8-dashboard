import json

import requests
from bs4 import BeautifulSoup


def parse_documentation(doc_url, tag="div", attributes={}):
    code_descriptions = dict()
    response = requests.get(doc_url)

    soup = BeautifulSoup(response.content, 'html.parser')

    error_codes_div = soup.find_all(tag, attributes)[0]

    row_tags = error_codes_div.find_all("tr")

    for tag in row_tags:
        columns = tag.find_all("td")
        if len(columns) > 0:
            error_code = columns[0].text
            error_code = error_code.strip()
            if len(error_code) > 0:
                if len(error_code) > 4:
                    error_code = error_code[:5]
                code_descriptions[error_code] = columns[1].text
    return code_descriptions


# Parse flake8 codes's description from documentation
code_description = parse_documentation('http://flake8.pycqa.org/en/2.5.5/warnings.html',
                                       attributes={"class": "section", "id": "warning-error-codes"})

# Parse pep8 codes's description from documentation
code_description.update(parse_documentation('https://pep8.readthedocs.io/en/latest/intro.html',
                                            attributes={"class": "section", "id": "error-codes"}))

# Parse pep8-naming codes's description from documentation
code_description.update(parse_documentation('https://github.com/PyCQA/pep8-naming/blob/master/README.rst',
                                            tag="table"))

# Mccabe codes? TODO

code_description.update(
    {
        'E': 'pep8',
        'W': 'pep8',
        'F': 'pyFlakes',
        'C': 'McCabe complexity',
        'C9': 'McCabe complexity',
        'N8': 'Naming convention',
    }
)

code_description = json.dumps(code_description, indent=4, sort_keys=True)

with open("../flake8_reporters/code_description.json", 'w') as file:
    file.write(code_description)
