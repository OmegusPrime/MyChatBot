from FileLoader import FileLoader

class DataPipeline:
    def __init__(self, directory):
        self.loader = FileLoader(directory)

    def run(self):
        texts = self.loader.load_all()
        return texts if isinstance(texts, list) else [texts]