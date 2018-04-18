# Predicting Restaurant Health Inspection Scores with Yelp Reviews and Ratings

## Desciption:
For small, local restaurants, maintaining a high health inspection rating is critical to reputation. However, inspections are random, and owners are unable to know when inspectors will visit. Therefore, it's important for owners to have a tool to estimate how their restaurant will perform at the next inspection. Conversely, health inspectors can use data to target restaurants that could be at risk for low health ratings. This project utilizes Yelp reviews and ratings to predict how a restaurant will perform.

## Research Question(s) or Objective(s):
- How will a restaurant perform on its next health inspection?

## Similar Work in the Field:
- [UCSD Research](https://cseweb.ucsd.edu/~jmcauley/cse255/reports/fa15/036.pdf)
- [Yelp Blog Post](https://engineeringblog.yelp.com/2015/04/data-science-contest-keeping-it-fresh-predict-restaurant-health-scores.html)
- Additionally, there are many news articles from 2013 that mentions Yelp adding health inspection score to their platform, but it is unable to be seen from the app and API

## Final Presentation:
- Presentation (Deck)
- Website (Time permitting)
- Interactive Restaurant Map (Time Permitting)

## Data Sources:
- Health Inspection Scores (24.9k scores): [data.austintexas.gov](https://data.austintexas.gov/Health-and-Community-Services/Restaurant-Inspection-Scores/ecmv-9xxi)
- Reviews and Ratings (11 million reviews that will be filtered to Austin Restaurants): [Yelp Dataset](Yelp.com/dataset)
- Both hosted on a MySQL database

## Built With
- [Python](https://docs.python.org/3/)
- [MySQL](https://www.mysql.com/)

## Challenges:
- Health inspections are way more frequent than I thought. I'm not sure Yelp reviews are frequent enough to predict each and every score. I'll likely have to bin data in weird ways
- Merging separated data sources on names that are not unique
- Currently, I'm having issues opening the SQL dump from Yelp, but I can solve this quickly
- Pulling out cleanliness and hygiene-related terms, as I suspect they are not as frequent as needed for NLP analysis

## Related Areas to Explore
- Improved restaurant recommender based on power users or users that have similar ratings to you

## Author
*Dave Spitzfaden*
