import pandas as pd
from pathlib import Path
import PyPDF2

class FileLoader:
    def __init__(self, directory):
        self.directory = Path(directory)

    def load_all(self):
        texts = []
        for file in self.directory.glob("*"):
            if file.suffix == ".txt":
                texts.append(self._load_txt(file))
            elif file.suffix == ".pdf":
                texts.append(self._load_pdf(file))
            elif file.suffix == ".csv":
                texts.append(self._load_csv(file))
        return texts

    def _load_txt(self, file):
        with open(file, "r", encoding="utf-8") as f:
            return f.read()

    def _load_csv(self, file):
        df = pd.read_csv(file)
        return " ".join(df.astype(str).values.flatten())

    def _load_pdf(self, file):
        text = ""
        with open(file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
        return text