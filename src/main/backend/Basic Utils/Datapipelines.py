import FileLoader
class Datapipelines:
    def __inti__(self, dir):
        self.loader = FileLoader(dir)
    def run(self):
        texts = self.loader.load_all()
        return texts