import requests

def info(query):
    try:
        r = requests.get(f"https://api.crossref.org/journals?query={query}", timeout=10)
        items = r.json().get("message", {}).get("items", [])
        if items:
            print(f"Query: {query} | Found: {items[0].get('title')} | Publisher: {items[0].get('publisher')}")
        else:
            print(f"Query: {query} | Not found")
    except:
        print(f"Error querying {query}")

print("--- Web Check ---")
for url in ["https://www.sciencedirect.com/journal/next-materials", "https://www.cell.com/joule/home"]:
    try:
        r = requests.head(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5, allow_redirects=True)
        print(f"{url}: {r.status_code}")
    except:
        print(f"{url}: Connection error")

print("\n--- Crossref Check ---")
info("Next Materials")
info("Watt")
