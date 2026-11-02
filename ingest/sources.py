"""What the database is built from. Change scope here and nowhere else."""

ARCHIVE_PAGE = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
BASE_URL = "https://files.consumerfinance.gov/f/documents/"

# Files 15 to 21 of the archive cover January to August 2026.
FILES = [
    "CCDB_Export_15_January_2026_through_February_2026.zip",
    "CCDB_Export_16_March_2026.zip",
    "CCDB_Export_17_April_2026.zip",
    "CCDB_Export_18_May_2026.zip",
    "CCDB_Export_19_June_2026.zip",
    "CCDB_Export_20_July_2026.zip",
    "CCDB_Export_21_August_2026.zip",
]

PRODUCTS = ("Credit card", "Checking or savings account")
DATE_FROM = "2026-01-01"
DATE_TO = "2026-08-31"

RAW_DIR = "data/raw"
MANIFEST = "data/manifest.json"
DB_PATH = "data/complaints.sqlite"


def url_for(name):
    return BASE_URL + name
