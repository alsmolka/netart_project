# Weed Out the Lies
## Fake News Detection NetArt Project 
[description]
The recordings from the project can be viewed in the slides (https://docs.google.com/presentation/d/10QYdX8nj5HcqfQ5WnXKZIFtkCmautGfwxqop-DTbjuY/edit#slide=id.p)

## Repository content
The repository contains the code for training and running predictions on the 

You have to first download and unzip the models from https://drive.google.com/drive/folders/1trCsMiz8JGNQVLeULVq0dYdNqLCCmQjT?usp=sharing. The requirements are in "requirements.txt"
Files with the data from the crawler have to be in csv, English must have "title" column, Taiwanese and Chinese must have "content" column.

To run the prediction see the example scripts. The first argument is the news source ("en", "ch" or ""tw"). Second argument is the csv file with news.

The program will save the results in the main directory in a json file (number of real and fake news and ratio)
