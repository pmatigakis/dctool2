from bs4 import BeautifulSoup


def process_web_page(content):
    soup = BeautifulSoup(content, 'html.parser')
    [s.extract() for s in soup.findAll("script")]
    [s.extract() for s in soup.findAll("style")]
    return soup.get_text()
