LOG_FILE='../model/mountaincar/default/log.txt'
# LOG_FILE='../model/cartpole/default/log.txt'
last_line_num=0
while True:
    with open(LOG_FILE,'r') as file: lines = file.readlines()
    if len(lines) != last_line_num:
        print_lines = lines[-(len(lines)-last_line_num):]
        last_line_num = len(lines)
        for line in print_lines: print(line.strip())
