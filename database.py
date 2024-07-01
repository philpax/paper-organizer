import json
import sqlite3
import os
import sys
import arxiv
import argparse
import re
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found.")
        sys.exit(1)


def is_valid_arxiv_id(arxiv_id):
    pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
    return re.match(pattern, arxiv_id) is not None


class ArXivOrganizer:
    def __init__(self, unsorted_path: str):
        self.unsorted_path = unsorted_path
        self.db_path = "papers.db"
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()
        self.client = arxiv.Client()

        self._init_db()

    def _init_db(self):
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS papers
                        (id TEXT PRIMARY KEY, title TEXT, abstract TEXT, file_path TEXT)"""
        )
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS authors
                        (id INTEGER PRIMARY KEY, name TEXT UNIQUE)"""
        )
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS categories
                        (id INTEGER PRIMARY KEY, name TEXT UNIQUE)"""
        )
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS paper_authors
                        (paper_id TEXT, author_id INTEGER,
                        FOREIGN KEY(paper_id) REFERENCES papers(id),
                        FOREIGN KEY(author_id) REFERENCES authors(id),
                        PRIMARY KEY(paper_id, author_id))"""
        )
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS paper_categories
                        (paper_id TEXT, category_id INTEGER,
                        FOREIGN KEY(paper_id) REFERENCES papers(id),
                        FOREIGN KEY(category_id) REFERENCES categories(id),
                        PRIMARY KEY(paper_id, category_id))"""
        )
        self.conn.commit()

    def get_paper_details(self, arxiv_id):
        search = arxiv.Search(id_list=[arxiv_id])
        return next(self.client.results(search))

    def add_paper(self, file_path):
        try:
            filename = os.path.splitext(os.path.basename(file_path))[0]
            arxiv_id = filename.split(" - ")[0]
            if not is_valid_arxiv_id(arxiv_id):
                print(f"Skipping {filename}: Not a valid arXiv ID format.")
                return

            # Check if paper already exists
            self.c.execute("SELECT id FROM papers WHERE id = ?", (arxiv_id,))
            if self.c.fetchone() is not None:
                print(f"Paper {arxiv_id} already exists in the database. Skipping.")
                return

            # Fetch metadata from arXiv
            paper = self.get_paper_details(arxiv_id)

            # Insert paper info into database
            self.c.execute(
                "INSERT INTO papers (id, title, abstract, file_path) VALUES (?, ?, ?, ?)",
                (arxiv_id, paper.title, paper.summary, file_path),
            )

            # Add authors
            for author in paper.authors:
                self.c.execute(
                    "INSERT OR IGNORE INTO authors (name) VALUES (?)", (author.name,)
                )
                self.c.execute("SELECT id FROM authors WHERE name = ?", (author.name,))
                author_id = self.c.fetchone()[0]
                self.c.execute(
                    "INSERT INTO paper_authors (paper_id, author_id) VALUES (?, ?)",
                    (arxiv_id, author_id),
                )

            # Add categories
            for category in paper.categories:
                self.c.execute(
                    "INSERT OR IGNORE INTO categories (name) VALUES (?)", (category,)
                )
                self.c.execute("SELECT id FROM categories WHERE name = ?", (category,))
                category_id = self.c.fetchone()[0]
                self.c.execute(
                    "INSERT INTO paper_categories (paper_id, category_id) VALUES (?, ?)",
                    (arxiv_id, category_id),
                )

            self.conn.commit()
            print(f"{arxiv_id}: {paper.title}")
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding paper: {str(e)}")

    def add_folder(self, folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=True):
            # Modify dirs in-place to exclude 'Unsorted' from further traversal
            dirs[:] = [d for d in dirs if d != "Unsorted"]

            for file in files:
                if file.endswith(".pdf"):
                    self.add_paper(os.path.join(root, file))

    def remove_paper(self, arxiv_id):
        try:
            # Check if paper exists
            self.c.execute("SELECT id FROM papers WHERE id = ?", (arxiv_id,))
            if self.c.fetchone() is None:
                print(f"Paper {arxiv_id} not found in the database.")
                return

            # Remove from all tables
            self.c.execute("DELETE FROM paper_authors WHERE paper_id = ?", (arxiv_id,))
            self.c.execute(
                "DELETE FROM paper_categories WHERE paper_id = ?", (arxiv_id,)
            )
            self.c.execute("DELETE FROM papers WHERE id = ?", (arxiv_id,))

            self.conn.commit()
            print(f"Paper {arxiv_id} removed successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error removing paper: {str(e)}")

    def search(self, query, limit=5):
        try:
            self.c.execute(
                """
                SELECT DISTINCT p.id, p.title, GROUP_CONCAT(DISTINCT a.name) as authors
                FROM papers p
                LEFT JOIN paper_authors pa ON p.id = pa.paper_id
                LEFT JOIN authors a ON pa.author_id = a.id
                LEFT JOIN paper_categories pc ON p.id = pc.paper_id
                LEFT JOIN categories c ON pc.category_id = c.id
                WHERE p.title LIKE ? OR p.abstract LIKE ? OR a.name LIKE ? OR c.name LIKE ?
                GROUP BY p.id
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit),
            )
            results = self.c.fetchall()
            return results
        except Exception as e:
            print(f"Error searching papers: {str(e)}")
            return []

    def show_paper(self, arxiv_id):
        try:
            self.c.execute(
                """
                SELECT p.title, p.abstract, p.file_path, GROUP_CONCAT(a.name, '; ') as authors
                FROM papers p
                LEFT JOIN paper_authors pa ON p.id = pa.paper_id
                LEFT JOIN authors a ON pa.author_id = a.id
                WHERE p.id = ?
                GROUP BY p.id
            """,
                (arxiv_id,),
            )
            result = self.c.fetchone()

            if result:
                title, abstract, file_path, authors = result
                cleaned_file_path = os.path.abspath(file_path).replace(" ", "%20")
                print(f"ID: {arxiv_id}")
                print(f"Title: {title}")
                print(f"Authors: {authors}")
                print(f"Abstract: {abstract}")
                print(f"File Path: file://{cleaned_file_path}")
            else:
                print(f"No paper found with ID: {arxiv_id}")
        except Exception as e:
            print(f"Error showing paper: {str(e)}")

    def download_paper(self, paper_id):
        if not is_valid_arxiv_id(paper_id):
            print("Invalid arXiv ID format: {paper_id}")
            return

        paper = self.get_paper_details(paper_id)

        title = (
            paper.title.replace(": ", " - ")
            .replace("? ", " - ")
            .replace("?", "")
            .replace("/", "-")
        )
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        filename = f"{paper_id} - {title}.pdf"
        full_path = os.path.join(self.unsorted_path, filename)

        try:
            pdf_response = requests.get(url, timeout=30)
            pdf_response.raise_for_status()
            with open(full_path, "wb") as f:
                f.write(pdf_response.content)
            print(f"Successfully downloaded: {filename}")
        except requests.RequestException as e:
            print(f"Failed to download {filename}: {str(e)}")

    def validate_date(self, date_string):
        try:
            datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def fetch_daily_papers(self, date):
        if not self.validate_date(date):
            print("Invalid date format. Please use YYYY-MM-DD.")
            return

        url = f"https://huggingface.co/api/daily_papers?date={date}"
        response = requests.get(url)
        if response.status_code == 200:
            papers = response.json()
        else:
            print(f"Error fetching data: {response.status_code}")
            return

        sorted_papers = sorted(papers, key=lambda x: -x["paper"]["upvotes"])
        processed_paper_ids = [paper["paper"]["id"] for paper in sorted_papers]

        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.download_paper, paper_id)
                for paper_id in processed_paper_ids
            ]
            for future in as_completed(futures):
                _ = future.result()

    def close(self):
        self.conn.close()


def main():
    config = load_config()
    unsorted_path = config.get("unsorted_path")

    parser = argparse.ArgumentParser(description="arXiv Paper Organizer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add paper
    add_parser = subparsers.add_parser("add", help="Add a paper to the database")
    add_parser.add_argument("file_path", help="Path to the PDF file")

    # Add folder
    add_folder_parser = subparsers.add_parser(
        "add-folder", help="Add all papers in a folder to the database"
    )
    add_folder_parser.add_argument(
        "folder_path", help="Path to the folder containing PDF files"
    )

    # Remove paper
    remove_parser = subparsers.add_parser(
        "remove", help="Remove a paper from the database"
    )
    remove_parser.add_argument("paper_id", help="ID of the paper to remove")

    # Download paper
    download_parser = subparsers.add_parser(
        "download", help="Download a paper from arXiv"
    )
    download_parser.add_argument("paper_id", help="ID of the paper to download")

    # Search
    search_parser = subparsers.add_parser("search", help="Search for papers")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-l", "--limit", type=int, default=5, help="Number of results to return"
    )

    # Show paper
    show_parser = subparsers.add_parser("show", help="Show details of a specific paper")
    show_parser.add_argument("paper_id", help="ID of the paper to show")

    # Fetch daily papers
    fetch_parser = subparsers.add_parser(
        "fetch-daily-papers", help="Fetch and download daily papers from Hugging Face"
    )
    fetch_parser.add_argument("date", help="Date in YYYY-MM-DD format")

    args = parser.parse_args()

    organizer = ArXivOrganizer(unsorted_path)

    if args.command == "add":
        organizer.add_paper(args.file_path)
    elif args.command == "add-folder":
        organizer.add_folder(args.folder_path)
    elif args.command == "remove":
        organizer.remove_paper(args.paper_id)
    elif args.command == "download":
        organizer.download_paper(args.paper_id)
    elif args.command == "search":
        results = organizer.search(args.query, args.limit)
        print(f"Top {args.limit} results for '{args.query}':")
        for paper_id, title, authors in results:
            print(f"ID: {paper_id}")
            print(f"Title: {title}")
            print(f"Authors: {authors}")
            print("---")
    elif args.command == "show":
        organizer.show_paper(args.paper_id)
    elif args.command == "fetch-daily-papers":
        organizer.fetch_daily_papers(args.date)
    else:
        parser.print_help()

    organizer.close()


if __name__ == "__main__":
    main()
