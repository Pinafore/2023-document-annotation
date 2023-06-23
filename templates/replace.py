import os

def replace_text_in_file(file_path, text_to_search, replacement_text):
    with open(file_path, 'r') as file:
        filedata = file.read()
        
    new_data = filedata.replace(text_to_search, replacement_text)

    if new_data != filedata:
        with open(file_path, 'w') as file:
            file.write(new_data)
        
def main():
    text_to_search = '<a class="active" href="{{url_for(\'home_page\', name=session[\'user_id\'])}}"'
    replacement_text = '<a class="active" href="{{url_for(\'home_page\', name=session[\'name\'])}}"'
    

    for subdir, dirs, files in os.walk('.'):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".html") or filepath.endswith(".py"):
                replace_text_in_file(filepath, text_to_search, replacement_text)

if __name__ == "__main__":
    main()
