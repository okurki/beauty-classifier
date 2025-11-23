import requests
import os
import time
from tqdm import tqdm


def download_wikimedia_images(
    filenames: list[str],
    download_folder="datasets/celebrities_pretty",
    debug: bool = False,
):
    agent_header = {"User-Agent": "CringeBot/1.0 (https://github.com/sasaSilver)"}
    pbar = tqdm(filenames)
    for i, name in enumerate(pbar):
        search_name = " ".join(
            map(lambda name: name.capitalize(), name.replace(".jpg", "").split("_"))
        )
        pbar.set_description(f"Processing {i + 1}/{len(filenames)}: {name}")
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": search_name,
            "srnamespace": "6",
        }
        res = requests.get(url, params=params, headers=agent_header)
        res.raise_for_status()
        response = res.json()
        debug and print(f"Got response: {response}")
        file_title = response["query"]["search"][0]["title"]
        debug and print(f"Got file title: {file_title}")
        # Get file URL
        params = {
            "action": "query",
            "format": "json",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
        }
        res = requests.get(url, params=params, headers=agent_header)
        res.raise_for_status()
        response = res.json()
        page = next(iter(response["query"]["pages"].values()))
        if "imageinfo" not in page:
            raise ValueError(f"No image info in page: {page}")
        img_url = page["imageinfo"][0]["url"]
        debug and print(f"Got image URL: {img_url}")
        # Download image
        img_data = requests.get(img_url, headers=agent_header).content
        filename = f"{download_folder}/{name}.jpg"
        with open(filename, "wb") as f:
            written = f.write(img_data)
            if written == 0:
                raise ValueError("Failed to write image data")
        debug and print(f"Downloaded from Wikimedia: {filename}")
        time.sleep(0.5)  # Rate limiting


DEBUG = False

to_download = set(os.listdir("datasets/open_famous_people_faces"))
already_downloaded = set(
    filename.replace(".jpg", "")
    for filename in os.listdir("datasets/celebrities_pretty")
    if filename.endswith(".jpg")
)
filenames = to_download.difference(already_downloaded)
filenames.remove("classes.json")  # not needed

print(
    f"Already downloaded {len(already_downloaded)} images",
    f"Total to download {len(to_download)} images",
    f"Downloading {len(filenames)} images",
    sep="\n",
)
download_wikimedia_images(filenames)
