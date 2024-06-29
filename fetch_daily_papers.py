import sys
import requests
from datetime import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def validate_date(date_string):
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found.")
        sys.exit(1)


def download_paper(paper, unsorted_path):
    paper_id = paper["paper_id"]
    title = paper["title"]
    title = (
        title.replace(": ", " - ")
        .replace("? ", " - ")
        .replace("?", "")
        .replace("/", "-")
    )
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    filename = f"{paper_id} - {title}.pdf"
    full_path = os.path.join(unsorted_path, filename)

    try:
        pdf_response = requests.get(url, timeout=30)
        pdf_response.raise_for_status()
        with open(full_path, "wb") as f:
            f.write(pdf_response.content)
        return f"Successfully downloaded: {filename}"
    except requests.RequestException as e:
        return f"Failed to download {filename}: {str(e)}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_daily_papers.py YYYY-MM-DD")
        sys.exit(1)

    date = sys.argv[1]
    if not validate_date(date):
        print("Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    config = load_config()
    unsorted_path = config.get("unsorted_path")

    url = f"https://huggingface.co/api/daily_papers?date={date}"
    response = requests.get(url)
    if response.status_code == 200:
        papers = response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        sys.exit(1)

    sorted_papers = sorted(papers, key=lambda x: -x["paper"]["upvotes"])
    processed_papers = [
        {"paper_id": paper["paper"]["id"], "title": paper["title"]}
        for paper in sorted_papers
    ]

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_paper = {
            executor.submit(download_paper, paper, unsorted_path): paper
            for paper in processed_papers
        }
        for future in as_completed(future_to_paper):
            print(future.result())


if __name__ == "__main__":
    main()
