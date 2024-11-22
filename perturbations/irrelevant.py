import requests
import wikipedia
from bs4 import BeautifulSoup

WIKI_DOMAIN = "https://en.wikipedia.org"
NUMBER_OF_PAGES = 20


def fetch_links(link):
    req = requests.get(link)
    req.raise_for_status()
    html_soup = BeautifulSoup(req.text, "html.parser")
    for a_tag in html_soup.find_all("a"):
        yield a_tag.get("href")


def fetch_wiki_content(link):
    response = requests.get(link)
    if response is not None:
        html = BeautifulSoup(response.text, "html.parser")
        title = html.select("#firstHeading")[0].text
        paragraphs = html.select("p")
        intro = "".join([para.text for para in paragraphs])
        return intro


def wikicrawler(start):
    queue = [start]
    seen = {start}
    downloaded = 0
    while queue:
        print("Downloaded {} pages".format(downloaded))
        if downloaded > NUMBER_OF_PAGES:
            break

        link = queue.pop(0)
        content = fetch_wiki_content(link)
        downloaded += 1
        with open(
            "data/irrelevant/{}.txt".format(
                link.split("wiki/")[-1].replace("/", "_").replace(":", "_")
            ),
            "w",
        ) as file:
            file.write(content)

        page_links = 0
        try:
            for i, href in enumerate(fetch_links(link)):
                if href is not None and "/wiki" == href[:5]:
                    if i < 360:
                        continue
                    if page_links > 5:
                        break

                    new_link = WIKI_DOMAIN + href
                    if new_link not in seen:
                        queue.append(new_link)
                        seen.add(new_link)
                        page_links += 1
        except Exception:
            continue


# wikicrawler("https://en.wikipedia.org/wiki/Barack_Obama")
# wikicrawler("https://en.wikipedia.org/wiki/Taylor_Swift")
# wikicrawler("https://en.wikipedia.org/wiki/LeBron_James")
# wikicrawler("https://en.wikipedia.org/wiki/Greece")
wikicrawler("https://en.wikipedia.org/wiki/Photosynthesis")
