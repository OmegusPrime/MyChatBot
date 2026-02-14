import os 
import pandas as pd   
import numpy as np       
from pathlib import Path   
import re 
import PyPDF2 
class FileLoader:
    def __init__(slef,directory):
        self.directory = Path(directory)
    def load_all(self):
        texts=[]
        for file in self.directory.glob("*"):
            if file.suffix == ".txt":
                texts.append(self._load_txt(file))
            elif file.suffix == ".pdf":
                texts.append(self._load_pdf(file))
            elif file.suffix == ".csv":
                texts.apend(self._load_csv(file))
        return texts
    def _load_txt(self,file):
        with open(file, "r",encoding = "utf-8") as f:
            return f.read()
    def _load_csv(self,file):
        df = pdf.read_csv(file)
        return " ".join(df.astype(str).values.flatten())
    def _load_pdf(self,file):
        text = ""
        with open(file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for pages in reader.pages:
                text+=pages.extract_text()+" "
        return text
    