# netart_project
You have to first download and unzip the models from https://drive.google.com/drive/folders/1trCsMiz8JGNQVLeULVq0dYdNqLCCmQjT?usp=sharing.
Files with the data from the crawler have to be in csv, English must have "title" column, Taiwanese and Chinese must have "content" column.

To run the prediction see the example scripts. The first argument is the news source ("en", "ch" or ""tw"). Second argument is the csv file with news.

The program will save the results in the main directory in a json file (number of real and fake news and ratio)
