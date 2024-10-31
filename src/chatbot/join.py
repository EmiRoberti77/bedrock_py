EXIT = 'exit'
GOOD_BYE = 'good bye'
history = []

def getHistory():
    return "\n".join(history)

while True:
    user_input = input('>')
    if user_input.lower() == EXIT:
        print(GOOD_BYE)
        break
    
    history.append("User: " + user_input)
    print(getHistory())