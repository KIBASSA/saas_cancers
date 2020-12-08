
class AzureExperimentMoq(object):
    def __init__(self):
        self.runs = []

    def get_runs(self):
        return self.runs
        
class WeorkspaceMoq(object):
    def __init__(self):
        print("WeorkspaceMoq")

class ExperimentMoq(object):
    def __init__(self):
        self.workspace = WeorkspaceMoq()

class AzureMLRunMoq(object):
    def __init__(self, parent):
        self.parent = parent

        self.experiment = ExperimentMoq()

        self.children = []
        self.json_data = {}
        self.tags = {}
        self.metrics = {}

    def get_children(self):
        return self.children
    
    def get_details(self):
        return self.json_data

    def get_tags(self):
        return self.tags
    
    def get_metrics(self):
        return self.metrics
    
    def tag(self, title, value):
        print("title :", title)
        print("value :", value)

    def log(self, title, value):
        print("title :", title)
        print("value :", value)

    def log_image(self, title, plot):
        print("tile:", title)
        print("plot :", plot)
        id = Helper.generate()
        if self.temp_dir:
            plot.savefig(os.path.join(self.temp_dir,"{0}_{1}.png".format(title, id)))
    
    def upload_file(self, name, path_or_stream):
        print("name :", name)
        print("path_or_stream :", path_or_stream)
    
    def set_temp_dir(self, temp_dir):
        self.temp_dir = temp_dir
    
    def register_model(self, model_name,tags,model_path):
        print("model_name : ", model_name)
        print("model_path : ", model_path)