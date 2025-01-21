import subprocess

def select_box(r,c):
    print(f"Selecting box {r} {c}")
    a=r*120+70
    b=c*120+420
    command = f"adb shell input tap {a} {b}"
    res = subprocess.run(command, shell=True, text=True, capture_output=True)
    if res.stderr:
        print(f"Error: {res.stderr}")
    else:
        print(f"Location {a} {b} is selected")
        pass

while True:
    r = int(input("Enter row: "))
    c = int(input("Enter column: "))
    select_box(r,c)