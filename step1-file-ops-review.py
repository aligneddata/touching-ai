import os

print("Current dir is: " + os.getcwd())

# Open the file in read mode
file = open("step1-file-ops-review.data.txt", "r")  # os.getcwd(), os.chdir()

# Read each line one by one
for line in file:
    print(f"Line content [{line.strip()}]")  # .strip() to remove newline characters

# Close the file
file.close()