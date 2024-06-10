import os, sys
import asyncio
import aiofiles
import tiktoken

from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Transform the observation file into a list of observations
async def chunk_observations(filename, max_tokens=12000):
  chunks = []
  chunk = ""
  current_tokens = 0

  async with aiofiles.open(filename, "r", encoding="utf8") as file:
    contents = await file.read()
    observations = contents.split("\n")
    for observation in observations:
      tokens = len(enc.encode(observation))
      if current_tokens + tokens < max_tokens:
        chunk += observation
        current_tokens += tokens
      else:
        chunks.append(chunk)
        chunk = observation
        current_tokens = tokens
    chunks.append(chunk)
  return chunks

# Generate notes from a list of observations
async def generate_notes(name, observations):
  notes = "- Name: {name}".format(name=name)
  for observation in observations:
    prompt = f"""
      Review the observations about {name} and collect notes to create a notes for them.
      - DO NOT REWRITE the existing notes, add new notes.
      - Follow the single-line bullet point format provided in the notes.
      - Be specific about {name}'s interests, knowledge, relationships, and interactions.
      - Incorporate catchphrases, habits, comedic style, and character dynamics.
      - Include any new information or details about {name}.

      Current Notes:
      ```
      {notes}
      ```

      Observations:
    """.format(notes=notes, name=name)
    completion = await client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=1,
      max_tokens=4096,
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": observation}
      ]
    )
    context = completion.choices[0].message.content
    async with aiofiles.open(f"./notes/{name}.txt", "w+", encoding="utf8") as file:
      await file.write(context)

async def main(filename):
  observations = await chunk_observations(filename)
  name = os.path.basename(filename).split(".")[0]

  await generate_notes(name, observations)

if __name__ == "__main__":
  asyncio.run(main(sys.argv[1]))