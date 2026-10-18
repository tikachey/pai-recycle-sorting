"""Check that the Gemini API key works and list robotics models (never prints the key)."""

import argparse
import os

from dotenv import load_dotenv
from google import genai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()
    load_dotenv(args.env_file)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    names = [m.name for m in client.models.list()]
    print(f"{len(names)} models visible")
    for n in names:
        if "robotics" in n or "flash" in n:
            print(" ", n)


if __name__ == "__main__":
    main()
