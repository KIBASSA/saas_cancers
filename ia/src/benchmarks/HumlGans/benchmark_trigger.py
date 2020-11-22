import os
import shutil
from os.path import isfile, join
import sys
import subprocess
class BenchmarkTrigger(object):
    def _get_path_files(self, root, exclude_files=[]):
        """[summary]
        Arguments:
            root {[type]} -- [description]

        Keyword Arguments:
            exclude_files {list} -- [description] (default: {[]})

        Returns:
            [type] -- [description]
        """
        result = []
        for root, _, files in os.walk(root):
            for filename in files:
                filepath = join(root, filename)
                dirname = os.path.dirname(filepath)

                to_continue = False
                for exclude in exclude_files:
                    if exclude in dirname:
                        to_continue = True
                
                for exclude in exclude_files:
                    if exclude in filename:
                        to_continue = True
                
                if to_continue == True:
                    continue

                result.append(filepath)
        return result

    def run(self):
        # Create a folder for the pipeline step files

        script_folder = 'benchmark_scripts'
        shutil.rmtree(script_folder, ignore_errors=True)
        os.makedirs(script_folder, exist_ok=True)
        try:
            #copy all necessary scripts
            files = self._get_path_files("..\\..\\", [os.path.basename(__file__), "data"])
            for f in files:
                shutil.copy(f, script_folder)

            #generated_config_file = "../../config.yaml"
            #shutil.copy(generated_config_file, script_folder)

            # start benchmark engine process
            os.chdir(script_folder)
            #subprocess.Popen(['x-terminal-emulator', '-e', 'python benchmark_engine.py'])
            os.system('python benchmark_engine.py')
            #result = subprocess.check_output('python benchmark_engine.py', shell=True)
            #print("result : ", result)
            #p = subprocess.Popen('python benchmark_engine.py', stdout=subprocess.PIPE, bufsize=1)
            #for line in iter(p.stdout.readline, b''):
            #    print(line)
            #p.stdout.close()
            #p.wait()

            #delete script_folder
            #os.chdir("../")
        finally:
            os.chdir("../")
            shutil.rmtree(script_folder, ignore_errors=True)
        

if __name__ == "__main__":
    trigger = BenchmarkTrigger()
    trigger.run()
