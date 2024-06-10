from datetime import datetime
import csv
import tiktoken

import os
import sys

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Loads the data from the CSV file, parses the dates and content, and returns a list of messages
def prepare_csv(filename):
  with open(filename, "r",  encoding="utf8") as file:
    reader = csv.reader(file)
    messages = []
    for row in reader:
      date = row[2]
      if date == "Date":
        continue
      date = datetime.strptime(date, "%m/%d/%Y %I:%M %p")
      messages.append((date, row[1], row[3],
                      len(enc.encode(",".join(row)))))
  messages.sort(key=lambda x: x[0])
  return messages

# convo messages by a maximum number of tokens
def convo_messages(messages, max_tokens = 16000):
  conversations = []
  current_convo = []
  current_tokens = 0
  for message in messages:
    if current_tokens + message[3] < max_tokens:
      current_convo.append(message)
      current_tokens += message[3]
    else:
      conversations.append(current_convo)
      current_convo = [message]
      current_tokens = message[3]
  conversations.append(current_convo)

  return conversations

# Save the convoed messages to a CSV file
def save_conversations(convo, filename):
  with open(filename, "w+", encoding="utf8", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "User", "Content"])
    for message in convo:
      writer.writerow((message[0].strftime("%m/%d/%Y %I:%M %p"), message[1], message[2]))

# Main function
def main():
  messages = prepare_csv(sys.argv[1])
  conversations = convo_messages(messages)

  files = len(os.listdir("./conversations"))

  for i, convo in enumerate(conversations):
    save_conversations(convo, "./conversations/convo-" + str(i + files) + ".csv")

if __name__ == "__main__":
  main()