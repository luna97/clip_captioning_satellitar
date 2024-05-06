# run the files in the following order:

def run_script(filename):
    with open(filename, 'r') as file:
        exec(file.read())


if __name__ == '__main__':
    print("Runnning training for clip model")
    run_script('caption.py')
    print("\nRunnning training for decoder")
    run_script('dataset.py')
    print("\nRunnning evaluation")
    run_script('main.py')