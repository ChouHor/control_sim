def get_text_file(filename):
    f = open(filename, "r")
    _content = f.read()
    f.close()
    return _content


all_content = get_text_file("all_content.txt")
contents = all_content.split(";start_character;")[1:]
num = len(contents)
for i in range(num - 1):
    if i < num - 2:
        content = contents[i]
        file_contents = content.split((";end_character;\n"))

    else:
        content = contents[i]
        file_contents = content.split((";end_character;\n"))
        file_contents[1] = file_contents[1] + contents[i + 1]
    file_name = "1" + file_contents[0]

    file_content = file_contents[1]

    fh = open(file_name, "w", encoding="utf-8")
    fh.write(file_content)
    fh.close()
