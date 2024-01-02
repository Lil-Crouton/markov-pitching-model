import os

# specify the path to your directory
directory_path = 'C:/Users/16086/Desktop/Baseball Tool/Event_Data/2022eve'  # e.g., 'C:/Users/YourName/Documents/'

# change the working directory to the specified path
os.chdir(directory_path)

# get all files from the directory
files = os.listdir()

# renaming each file
for file in files:
    # check the file extension and rename accordingly
    if file.endswith('.EVA') or file.endswith('.EVN'):
        # split the file name at the extension and get the first part
        root = os.path.splitext(file)[0]
        
        # add the new extension
        new_file_name = root + '.csv'
        
        # rename the file using os.rename()
        os.rename(file, new_file_name)

print("Renaming completed!")
