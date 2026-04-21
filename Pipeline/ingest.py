import os 
import pandas as pd   
import numpy as np       
from pathlib import Path   
import re 
import PyPDF2 
class Ingest:
    def __init__(self,path):
        self.path = Path(path)
    def ingest(self):
        for file_path in self.path.rglob("*"): 
            if not file_path.is_file():
                continue 
            if file_path.suffix == '.pdf':
                try:
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text_parts = []
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                        yield {'file': str(file_path), 'text': " ".join(text_parts)}
                except Exception as e:
                    print(f"PDF error: {file_path} -> {e}")
            elif file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        yield {'file': str(file_path), 'text': f.read()}
                except Exception as e:
                    print(f"TXT error: {file_path} -> {e}")

            elif file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(file_path)

                    for row in df.astype(str).values:  # faster
                        yield {
                            'file': str(file_path),
                            'text': ' '.join(row)
                        }

                except Exception as e:
                    print(f"CSV error: {file_path} -> {e}")
