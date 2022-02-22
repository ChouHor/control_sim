import os


def get_text_file(filename):
    f = open(filename, "r")
    _content = f.read()
    f.close()
    return _content


path = os.getcwd()
files = os.listdir(path)
files.sort(key=len)
all_content = ""
for file_name in files:
    if file_name[-3:] == ".py":
        start_character = ";start_character;" + file_name + ";end_character;\n"
        content = get_text_file(file_name)
        new_content = start_character + content
        all_content = all_content + new_content

fh = open("all_content.txt", "w", encoding="utf-8")
fh.write(all_content)
fh.close()
