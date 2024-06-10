import os, sys, csv
import asyncio
import aiofiles
import backoff

from openai import AsyncOpenAI, RateLimitError

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@backoff.on_exception(backoff.expo, RateLimitError)
async def process_convo(filename):
  names = []
  lines = []
  messages = ""

  # Load the chat log from the CSV file.
  async with aiofiles.open(filename, "r", encoding="utf8") as file:
    contents = await file.read()
    rows = contents.split("\n")
    for row in csv.reader(rows, delimiter=","):
      if len(row) < 3:
        continue
      if row[1] == "User":
        continue
      if row[1] not in names:
        names.append(row[1])
      lines.append(", ".join(row))

  messages = "\n".join(lines)

  prompt = """
  Generate a summary of the provided chat log. Include a summary of the conversation and any notable details, such as the date and time of the chat log, the users involved, and the topics discussed.
  """

  # Generate a summary of the chat log.
  completion = await client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": messages}
    ]
  )

  # Save the summary to a file.
  context = completion.choices[0].message.content
  filename = os.path.basename(filename)
  async with aiofiles.open("./contexts/" + filename + ".txt", "w+", encoding="utf8") as file:
    await file.write(context)

  # Generate character observations for each user in the chat log.
  tasks = []
  for name in names:
    tasks.append(generate_bio(name, messages))
  await asyncio.gather(*tasks)
  
async def generate_bio(name, messages):
  prompt = """
    Observe {name} in the chat log and generate a list of observations about them.
    - Include interests, knowledge, relationships, how they interact with others.
    - Draw conclusions about {name}'s life outside.
    - Use quotes from the chat log to support your observations.
    """.format(name=name)

  completion = await client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": messages}
      ]
    )
  
  context = completion.choices[0].message.content
  path = "./characters/" + name + ".txt"
  async with aiofiles.open(path, "a+", encoding="utf8") as file:
      await file.write(context + "\n\n")

async def main(start = 0):
  files = os.listdir("./conversations")

  # sort files by number in filename
  files = sorted(files, key=lambda x: int(x.split("-")[1].split(".")[0]))

  # Process each chat log in a separate thread.
  for file in files[start:]:
    await process_convo("./conversations/" + file)

if __name__ == "__main__":
  start = 0
  if len(sys.argv) > 1:
    start = int(sys.argv[1])
  asyncio.run(main(start))