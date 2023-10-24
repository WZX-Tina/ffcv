import argparse
def main(file_path):
    with open(file_path,'r') as file:
        for line in file:
            print(line,end = '')
if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'read')
    parser.add_argument('file',help = 'path to the file')
    args = parser.parse_args()
    main(args.file)
    
