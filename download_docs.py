import os
import requests
from bs4 import BeautifulSoup
import pdfkit

BASE_URL = "https://python.langchain.com"
START_URL = "https://python.langchain.com/docs/how_to/"

output_dir = "docs"
os.makedirs(output_dir, exist_ok=True)

response = requests.get(START_URL)
soup = BeautifulSoup(response.text, "html.parser")

links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.startswith("/docs/how_to/"):
        full_url = BASE_URL + href
        if full_url not in links:
            links.append(full_url)

for url in links:
    filename = url.strip("/").split("/")[-1] + ".pdf"
    output_path = os.path.join(output_dir, filename)

    print(f"Downloading: {url} -> {output_path}")
    try:
        pdfkit.from_url(url, output_path, options={
            "enable-local-file-access": "",
            "javascript-delay": "3000"
        })
    except Exception as e:
        print(f"Error while downloading {url}: {e}")
