# Final Proposal: Predicting Restaurant Health Inspection Scores from Yelp Reviews and Ratings

## Desciption:
For small, local restaurants, maintaining a high health inspection rating is critical to reputation. However, inspections are random and owners are unable to know when inspectors will visit. Therefore, it's important for owners to have a tool to estimate how their restaurant will perform at the next inspection. This project utilizes Yelp reviews and ratings to predict how a restaurant will perform.

## Research Question(s) or Objective(s):
- How will a restaurant perform on its next health inspection?

## Similar Work in the Field:
- I have found news stories that Yelp would start including health inspection scores, but it is not in the data dum and similarly not in the app interface, as far as I can tell.

## Final Presentation:
- Presentation and Website

## Data Sources:
- Health Inspection Scores: data.austintexas.gov
- Reviews and Ratings: Yelp.com/dataset
- Both hosted on a MySQL database

## Challenges:
- It seems like health inspections are way more frequent than I thought. I'm not sure Yelp reviews are frequent enough to predict each and every score. I'll likely have to bin data in weird ways.
- Merging separated data sources on names that are not unique.
- Currently, I'm having issues opening the SQL dump from Yelp.
- Pulling out cleanliness and hygiene-related terms.

## Related Areas to Explore
- Improved restaurant recommender based on power users or users that have similar ratings to you
- Conversely, health inspectors can use data to target restaurants that could be at risk
