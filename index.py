from dotenv import load_dotenv
load_dotenv()

import os
OPENAPI_KEY = os.environ.get('OPEN_API_ACCESS_KEY', '')

import openai
import pandas as pd

# Set up OpenAI API credentials
openai.api_key = OPENAPI_KEY

# Load CSV data into a Pandas DataFrame
df = pd.read_csv("sample.csv")

# Convert DataFrame to a list of dictionaries
data = df.to_dict("records")


"""
example user prompts for the "sample.csv" dataset:
1. explain to me how word problems are structured and how they may be analyzed
2. select any one of the word problems, recite the problem and solve it for me
3. what is the value of "median" and how did you arrive at that conclusion

answers from running the prompts:
1. Word problems are typically structured as follows: there is a problem to be solved, and a series of facts or clues that can be used to solve the problem. In order to solve the problem, you must first read the problem and identify the facts and clues that are given. Next, you must determine what operation(s) you will need to use in order to solve the problem. Finally, you must solve the problem and check your work.
2. If John has 3 apples and gives 1 to Jane, how many apples does John have left? John has 2 apples left.
3. The median is 30. The median is the value in the middle of the data set when the data is arranged in order from least to greatest.
"""

# Define the prompt that will be used by the language model
user_prompt = 'select any one of the word problems, recite the problem and solve it for me'
prompt = f"Given the following data, {user_prompt}. Here is the data:\n"

# Append the data to the prompt as a string
for row in data:
    prompt += str(row) + "\n"

# Call the GPT-3 API to generate text based on the prompt
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print the generated text
print(response.choices[0].text)
