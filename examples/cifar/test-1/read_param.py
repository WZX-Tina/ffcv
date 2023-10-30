import argparse
def main(file_path):

    with open(file_path,'r') as file:
        for line in file:
            data_list = line.split()
            converted_data = {}

# Iterate over each item in the data_list
            for item in data_list:
                key, value = item.split(":")
                if value.isdigit():
                    value = int(value)
                elif value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                converted_data[key] = value
            result = "training:\n"
            for key, value in converted_data.items():
                result += f"  {key}: {value}\n"

            print(result)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'read')
    parser.add_argument('file',help = 'path to the file')
    args = parser.parse_args()
    main(args.file)
    
