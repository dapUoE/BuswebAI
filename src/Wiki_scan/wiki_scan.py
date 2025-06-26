import requests
import time
import csv

def get_category_members(category, limit=500):
    base_url = "https://en.wikipedia.org/w/api.php"
    members = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "format": "json"
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = requests.get(base_url, params=params).json()
        members.extend(response["query"]["categorymembers"])

        if "continue" in response:
            cmcontinue = response["continue"]["cmcontinue"]
        else:
            break

        time.sleep(0.5)

    return members

def get_page_extract(title):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json"
    }
    response = requests.get(base_url, params=params).json()
    pages = response["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")

# Categories to extract
categories = [
    "Private_equity_firms_of_the_United_Kingdom",
    "Private_equity_firms_of_the_United_States",
    "Private_equity_firms_of_Europe"
]

all_firms = []
for cat in categories:
    print(f"Fetching category: {cat}")
    all_firms.extend(get_category_members(cat))

firm_titles = sorted(set(firm["title"] for firm in all_firms))

print(f"Total unique firms found: {len(firm_titles)}")

# Get descriptions
firm_data = []
for title in firm_titles:
    print(f"Getting info for: {title}")
    summary = get_page_extract(title)
    firm_data.append({"Name": title, "Description": summary})
    time.sleep(0.5)

# Save to CSV
csv_filename = "pe_firms_wikipedia.csv"
with open(csv_filename, mode="w", encoding="utf-8", newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Name", "Description"])
    writer.writeheader()
    writer.writerows(firm_data)

print(f"\n✅ Saved {len(firm_data)} firms to {csv_filename}")
