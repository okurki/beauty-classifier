import requests
import os
import time


def download_wikimedia_images(names: list[str], download_folder="datasets/celebrities"):
    agent_header = {"User-Agent": "CringeBot/1.0 (https://github.com/sasaSilver)"}
    for i, name in enumerate(names):
        print(f"Processing {i + 1}/{len(names)}: {name}")

        try:
            # Search Wikimedia Commons
            search_name = name.replace("_", " ")
            url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": f"{search_name} portrait",
                "srnamespace": "6",  # File namespace
            }

            response = requests.get(url, params=params, headers=agent_header).json()
            print(f"Got response: {response}")
            # Get image info for the first result
            if response["query"]["search"]:
                file_title = response["query"]["search"][0]["title"]
                print(f"Got file title: {file_title}")
                # Get file URL
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": file_title,
                    "prop": "imageinfo",
                    "iiprop": "url",
                }

                response = requests.get(url, params=params, headers=agent_header).json()
                page = next(iter(response["query"]["pages"].values()))

                if "imageinfo" in page:
                    img_url = page["imageinfo"][0]["url"]
                    print(f"Got image URL: {img_url}")
                    # Download image
                    img_data = requests.get(img_url, headers=agent_header).content
                    filename = f"{download_folder}/{name}.jpg"
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    print(f"  Downloaded from Wikimedia: {filename}")

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"Error processing {name}: {e}")


names = os.listdir("datasets/open_famous_people_faces")
download_wikimedia_images(names)
